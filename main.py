import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import datetime

app = FastAPI()

origins = [
    "http://localhost:3000",  # React app's origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory databases
users = {}
tokens = {}
bonds = {}
projects = {}
bond_counter = 1
user_counter = 1
current_user_id = None

dummy_user = {"id": 0, "name": "Sam", "balances": {}}
users[0] = dummy_user
current_user_id = 0

# Classes for ERC20Token and Bond operations
class ERC20Token:
    def __init__(self, id, name, symbol, total_supply):
        self.id = id
        self.name = name
        self.symbol = symbol
        self.total_supply = total_supply
        self.balances = {}
        self.allowances = {}

    def balanceOf(self, account):
        return self.balances.get(account, 0)

    def transfer(self, sender, receiver, amount):
        if self.balances.get(sender, 0) >= amount:
            self.balances[sender] -= amount
            self.balances[receiver] = self.balances.get(receiver, 0) + amount
            return True
        return False

    def approve(self, owner, spender, amount):
        if owner in self.balances:
            if owner not in self.allowances:
                self.allowances[owner] = {}
            self.allowances[owner][spender] = amount
            return True
        return False

    def transferFrom(self, spender, owner, receiver, amount):
        if self.allowances.get(owner, {}).get(spender, 0) >= amount and self.balances.get(owner, 0) >= amount:
            self.allowances[owner][spender] -= amount
            self.balances[owner] -= amount
            self.balances[receiver] = self.balances.get(receiver, 0) + amount
            return True
        return False

class BondContract(ERC20Token):
    def __init__(self, name, symbol, total_supply, issuer, interest_rate, maturity_date, payment_schedule, is_callable=False, is_convertible=False, penalty_rate=0, requires_full_sale=True, early_withdrawal=False):
        super().__init__(name, symbol, total_supply)
        self.issuer = issuer
        self.interest_rate = interest_rate
        self.maturity_date = maturity_date
        self.payment_schedule = payment_schedule
        self.is_callable = is_callable
        self.is_convertible = is_convertible
        self.penalty_rate = penalty_rate
        self.requires_full_sale = requires_full_sale
        self.early_withdrawal = early_withdrawal
        self.snapshots = []
        self.payments = {}
        self.payment_allocations = {}
        self.balances[issuer] = total_supply  # Issuer holds all bonds initially
        self.auction_price = total_supply  # Initialize auction price as total supply
        self.auction_end_time = None
        self.erc20_token = None  # Token used for payment (e.g., FRAX)
        self.bond_sold = False
        self.auction_active = False
        self.approved_payees = set([self.issuer])
        self.reentrancy_guard = False

    def only_issuer(func):
        def wrapper(self, *args, **kwargs):
            if self.issuer != args[0]:
                raise PermissionError("Only the issuer can call this function.")
            return func(self, *args, **kwargs)
        return wrapper

    def reentrancy_protection(func):
        def wrapper(self, *args, **kwargs):
            if self.reentrancy_guard:
                raise RuntimeError("Reentrancy detected.")
            self.reentrancy_guard = True
            result = func(self, *args, **kwargs)
            self.reentrancy_guard = False
            return result
        return wrapper

    def create_snapshot(self):
        snapshot = {holder: self.balances[holder] for holder in self.balances}
        self.snapshots.append(snapshot)
        return snapshot

    def allocate_payment(self, amount, payment_time):
        if payment_time not in self.payments:
            self.payments[payment_time] = 0
        self.payments[payment_time] += amount

    def claim_payment(self, holder, payment_time):
        if payment_time in self.payments and holder in self.snapshots[payment_time]:
            payment_per_token = self.payments[payment_time] / sum(self.snapshots[payment_time].values())
            holder_payment = self.snapshots[payment_time][holder] * payment_per_token
            self.payments[payment_time] -= holder_payment
            if self.erc20_token.balances[self.issuer] >= holder_payment:
                self.erc20_token.transfer(self.issuer, holder, holder_payment)
                return holder_payment
        return 0

    @only_issuer
    @reentrancy_protection
    def start_auction(self, issuer, initial_price, end_time, payment_token, manual_start=False):
        self.auction_price = initial_price
        self.auction_end_time = end_time if not manual_start else None
        self.erc20_token = payment_token  # Set the token used for payment
        self.auction_active = not manual_start

    @only_issuer
    @reentrancy_protection
    def manual_start_auction(self, issuer):
        if not self.auction_active and self.auction_end_time is None:
            self.auction_active = True
            self.auction_end_time = datetime.datetime.now() + datetime.timedelta(days=1)  # Example duration

    def bid_auction(self, bidder, bid_amount):
        if self.auction_active and (self.auction_end_time is None or datetime.datetime.now() < self.auction_end_time):
            if bid_amount >= self.auction_price and self.erc20_token.transferFrom(bidder, self.issuer, bid_amount):
                self.transfer(self.issuer, bidder, bid_amount)
                self.auction_price = bid_amount
                if not self.requires_full_sale or self.total_supply == 0:
                    self.bond_sold = True

    @only_issuer
    def end_auction(self, issuer):
        if datetime.datetime.now() >= self.auction_end_time or self.requires_full_sale and self.bond_sold:
            self.auction_active = False
            return True
        return False

    @only_issuer
    @reentrancy_protection
    def withdraw_funds(self, issuer, amount):
        if not self.requires_full_sale or self.bond_sold or self.early_withdrawal:
            if self.erc20_token.balances[self.issuer] >= amount:
                self.erc20_token.transfer(self.issuer, self.issuer, amount)
                return True
        return False

    def approve_payee(self, issuer, payee):
        if self.issuer == issuer:
            self.approved_payees.add(payee)

    def convert_to_shares(self, holder, conversion_rate, shares_contract):
        if self.is_convertible:
            num_shares = self.balances[holder] * conversion_rate
            shares_contract.transfer(self.issuer, holder, num_shares)
            self.balances[holder] = 0

    def call_bond(self, holder, call_price):
        if self.is_callable:
            self.transfer(holder, self.issuer, self.balances[holder])
            self.erc20_token.transferFrom(self.issuer, holder, call_price)

    def impose_penalty(self, holder, penalty_amount):
        self.balances[holder] -= penalty_amount

    def simulate_time_passage(self, days):
        self.auction_end_time -= datetime.timedelta(days=days)

# Pydantic models for API requests and responses
class User(BaseModel):
    id: int
    name: str
    balances: Dict[str, float] = {}

class Token(BaseModel):
    id: int
    name: str
    symbol: str
    total_supply: float
    balances: Dict[int, float] = {}

class PaymentScheduleItem(BaseModel):
    days: int
    months: int
    years: int
    amount: float

class Bond(BaseModel):
    id: int
    name: str
    symbol: str
    total_supply: float
    issuer: int
    interest_rate: float
    maturity_date: datetime.date
    payment_schedule: List[PaymentScheduleItem]
    is_callable: bool
    is_convertible: bool
    penalty_rate: float
    requires_full_sale: bool
    early_withdrawal: bool
    auction_price: float
    auction_end_time: Optional[datetime.datetime]
    bond_sold: bool
    auction_active: bool

class IssueBondRequest(BaseModel):
    name: str
    symbol: str
    total_supply: float
    interest_rate: float
    maturity_date: datetime.date
    payment_schedule: List[PaymentScheduleItem]
    is_callable: bool = False
    is_convertible: bool = False
    penalty_rate: float = 0
    requires_full_sale: bool = True
    early_withdrawal: bool = False

class CreateUserRequest(BaseModel):
    name: str

class CreateTokenRequest(BaseModel):
    name: str
    symbol: str
    total_supply: float

class AddFundsRequest(BaseModel):
    user_id: int
    token: str
    amount: float

class AuctionDuration(BaseModel):
    days: int
    hours: int

class AdjustmentDetails(BaseModel):
    intervalDays: int
    intervalHours: int
    amount: float
    rate: float

class BondDuration(BaseModel):
    years: int
    months: int
    days: int

class FixedPaymentInterval(BaseModel):
    days: int
    months: int
    years: int

class AuctionSchedule(BaseModel):
    auctionType: str
    auctionDuration: AuctionDuration
    auctionEndCondition: str
    adjustAutomatically: bool
    adjustmentDetails: AdjustmentDetails
    bondDuration: BondDuration
    repaymentType: str
    paymentSchedule: str
    fixedPaymentInterval: FixedPaymentInterval
    startAutomatically: bool
    startDate: str
    timezone: str

class BondDetails(BaseModel):
    title: str
    totalAmount: str
    infiniteTokens: bool
    tokens: str
    tokenPrice: str
    interestRate: str
    maxInterestRate: str
    minPrice: str
    requiresFullSale: bool
    latePenalty: str
    earlyRepayment: bool
    collateral: bool

class SaveDraftRequest(BaseModel):
    draft_id: Optional[int] = None
    project_info: Dict[str, str]
    bond_details: BondDetails
    auction_schedule: AuctionSchedule
    payment_schedule: List[PaymentScheduleItem]

class UserDetail(User):
    bonds_created: List[Bond] = []
    bonds_purchased: List[Bond] = []
    tokens_held: List[Token] = []
    draft_bonds: List[Dict] = []


# API Endpoints
@app.post("/users", response_model=User)
def create_user(user: CreateUserRequest):
    global user_counter
    user_id = user_counter
    user_counter += 1
    users[user_id] = {"id": user_id, "name": user.name, "balances": {}}
    return users[user_id]

@app.get("/users", response_model=List[User])
def get_users():
    return list(users.values())

@app.post("/users/{user_id}/add-funds")
def add_funds(request: AddFundsRequest):
    if request.user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    token_id = None
    for id, token in tokens.items():
        if token.symbol == request.token:
            token_id = id
            break

    if token_id is None:
        raise HTTPException(status_code=404, detail="Token not found")

    if request.token not in users[request.user_id]["balances"]:
        users[request.user_id]["balances"][request.token] = 0

    users[request.user_id]["balances"][request.token] += request.amount

    token = tokens[token_id]

    if request.user_id not in token.balances:
        token.balances[request.user_id] = 0

    token.balances[request.user_id] += request.amount

    return {"message": "Funds added successfully", "new_balance": users[request.user_id]["balances"][request.token]}

@app.get("/users/{user_id}", response_model=UserDetail)
def get_user(user_id: int):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users[user_id]
    
    # Retrieve bonds created by the user
    user_bonds = [bond for bond in bonds.values() if bond["issuer"] == user_id]
    user["bonds_created"] = user_bonds
    
    # Retrieve bonds purchased by the user
    user_bonds_purchased = [bond for bond in bonds.values() if user_id in bond["balances"] and bond["balances"][user_id] > 0]
    user["bonds_purchased"] = user_bonds_purchased
    
    # Retrieve tokens held by the user
    user_tokens = [token for token in tokens.values() if user_id in token.balances and token.balances[user_id] > 0]
    user["tokens_held"] = user_tokens
    
    # Retrieve draft bonds created by the user
    user_drafts = [draft for draft in projects.values() if draft["user_id"] == user_id and draft["is_draft"]]
    user["draft_bonds"] = user_drafts
    
    return user



@app.get("/current_user", response_model=User)
def get_current_user():
    global current_user_id
    if current_user_id is None:
        raise HTTPException(status_code=404, detail="No user is currently selected")
    return users[current_user_id]

@app.post("/users/{user_id}/switch")
def switch_user(user_id: int):
    global current_user_id
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    current_user_id = user_id
    return {"message": f"Switched to user {users[user_id]['name']}"}

@app.post("/tokens", response_model=Token)
def create_token(token: CreateTokenRequest):
    token_id = len(tokens) + 1
    tokens[token_id] = ERC20Token(token_id, token.name, token.symbol, token.total_supply)
    tokens[token_id].balances[token_id] = token.total_supply
    return {
        "id": token_id,
        "name": token.name,
        "symbol": token.symbol,
        "total_supply": token.total_supply,
        "balances": tokens[token_id].balances
    }
    return tokens[token_id]

@app.get("/tokens", response_model=List[Token])
def get_tokens():
    return list(tokens.values())

@app.post("/users/{user_id}/issue_bond", response_model=Bond)
def issue_bond(user_id: int, bond_request: IssueBondRequest):
    global bond_counter
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    bond_id = bond_counter
    bond_counter += 1
    bond = bond_request.dict()
    bond.update({
        "id": bond_id,
        "issuer": user_id,
        "auction_price": bond_request.total_supply,
        "auction_end_time": None,
        "bond_sold": False,
        "auction_active": False
    })
    bonds[bond_id] = bond
    return bond

@app.post("/users/{user_id}/save_draft", response_model=Dict)
def save_draft(user_id: int, draft_request: SaveDraftRequest):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    draft_id = draft_request.draft_id
    if draft_id and draft_id in projects and projects[draft_id]["user_id"] == user_id:
        # Update existing draft
        projects[draft_id].update({
            "project_info": draft_request.project_info,
            "bond_details": draft_request.bond_details,
            "auction_schedule": draft_request.auction_schedule,
            "payment_schedule": draft_request.payment_schedule,
            "is_draft": True
        })
        message = "Draft updated successfully"
    else:
        # Create new draft
        draft_id = len(projects) + 1
        projects[draft_id] = {
            "user_id": user_id,
            "draft_id": draft_id,
            "project_info": draft_request.project_info,
            "bond_details": draft_request.bond_details,
            "auction_schedule": draft_request.auction_schedule,
            "payment_schedule": draft_request.payment_schedule,
            "is_draft": True
        }
        message = "Draft saved successfully"

    return {"message": message, "draft_id": draft_id}

@app.delete("/users/{user_id}/delete_draft/{draft_id}", response_model=Dict)
def delete_draft(user_id: int, draft_id: int):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if draft_id not in projects or projects[draft_id]["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Draft not found or does not belong to the user")

    del projects[draft_id]
    return {"message": "Draft deleted successfully"}



@app.post("/users/{user_id}/start_auction/{bond_id}")
def start_auction(user_id: int, bond_id: int, initial_price: float, duration: int):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in bonds:
        raise HTTPException(status_code=404, detail="Bond not found")
    bond = bonds[bond_id]
    if bond["issuer"] != user_id:
        raise HTTPException(status_code=403, detail="Only the issuer can start the auction")
    bond["auction_price"] = initial_price
    bond["auction_end_time"] = datetime.datetime.now() + datetime.timedelta(days=duration)
    bond["auction_active"] = True
    return {"message": "Auction started"}

@app.post("/users/{user_id}/manual_start_auction/{bond_id}")
def manual_start_auction(user_id: int, bond_id: int):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in bonds:
        raise HTTPException(status_code=404, detail="Bond not found")
    bond = bonds[bond_id]
    if bond["issuer"] != user_id:
        raise HTTPException(status_code=403, detail="Only the issuer can manually start the auction")
    if not bond["auction_active"] and bond["auction_end_time"] is None:
        bond["auction_active"] = True
        bond["auction_end_time"] = datetime.datetime.now() + datetime.timedelta(days=1)  # Example duration
    return {"message": "Auction manually started"}

@app.post("/users/{user_id}/bid_auction/{bond_id}")
def bid_auction(user_id: int, bond_id: int, bid_amount: float):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in bonds:
        raise HTTPException(status_code=404, detail="Bond not found")
    bond = bonds[bond_id]
    if not bond["auction_active"]:
        raise HTTPException(status_code=400, detail="Auction is not active")
    if bond["auction_end_time"] is not None and datetime.datetime.now() >= bond["auction_end_time"]:
        bond["auction_active"] = False
        raise HTTPException(status_code=400, detail="Auction has ended")
    if bid_amount < bond["auction_price"]:
        raise HTTPException(status_code=400, detail="Bid amount is less than current auction price")
    
    token_id = None
    for id, token in tokens.items():
        if user_id in token.balances and token.balances[user_id] >= bid_amount:
            token_id = id
            break

    if token_id is None:
        raise HTTPException(status_code=400, detail="Insufficient balance")

    token = tokens[token_id]

    if not token.transfer(user_id, bond["issuer"], bid_amount):
        raise HTTPException(status_code=400, detail="Transfer failed")
    
    bond_contract = BondContract(**bond)
    bond_contract.transfer(bond["issuer"], user_id, bid_amount)
    bond["auction_price"] = bid_amount
    if not bond["requires_full_sale"] or bond_contract.total_supply == 0:
        bond["bond_sold"] = True
    return {"message": "Bid placed successfully"}

@app.post("/users/{user_id}/end_auction/{bond_id}")
def end_auction(user_id: int, bond_id: int):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in bonds:
        raise HTTPException(status_code=404, detail="Bond not found")
    bond = bonds[bond_id]
    if bond["issuer"] != user_id:
        raise HTTPException(status_code=403, detail="Only the issuer can end the auction")
    if datetime.datetime.now() >= bond["auction_end_time"] or bond["requires_full_sale"] and bond["bond_sold"]:
        bond["auction_active"] = False
        return {"message": "Auction ended"}
    return {"message": "Auction not yet ended"}

@app.post("/users/{user_id}/withdraw_funds/{bond_id}")
def withdraw_funds(user_id: int, bond_id: int, amount: float):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in bonds:
        raise HTTPException(status_code=404, detail="Bond not found")
    bond = bonds[bond_id]
    if bond["issuer"] != user_id:
        raise HTTPException(status_code=403, detail="Only the issuer can withdraw funds")
    bond_contract = BondContract(**bond)
    if not bond_contract.withdraw_funds(user_id, amount):
        raise HTTPException(status_code=400, detail="Withdrawal failed")
    return {"message": "Funds withdrawn successfully"}

@app.post("/users/{user_id}/allocate_payment/{bond_id}")
def allocate_payment(user_id: int, bond_id: int, amount: float, payment_time: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in bonds:
        raise HTTPException(status_code=404, detail="Bond not found")
    bond = bonds[bond_id]
    bond_contract = BondContract(**bond)
    bond_contract.allocate_payment(amount, payment_time)
    return {"message": "Payment allocated successfully"}

@app.post("/users/{user_id}/claim_payment/{bond_id}")
def claim_payment(user_id: int, bond_id: int, payment_time: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in bonds:
        raise HTTPException(status_code=404, detail="Bond not found")
    bond = bonds[bond_id]
    bond_contract = BondContract(**bond)
    payment_claimed = bond_contract.claim_payment(user_id, payment_time)
    if payment_claimed == 0:
        raise HTTPException(status_code=400, detail="No payment available for claim")
    return {"message": f"Payment of {payment_claimed} claimed successfully"}

@app.post("/users/{user_id}/simulate_time_passage/{bond_id}")
def simulate_time_passage(user_id: int, bond_id: int, days: int):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in bonds:
        raise HTTPException(status_code=404, detail="Bond not found")
    bond = bonds[bond_id]
    bond_contract = BondContract(**bond)
    bond_contract.simulate_time_passage(days)
    return {"message": "Time passage simulated successfully"}

@app.get("/users/{user_id}/bonds", response_model=List[Bond])
def get_user_bonds(user_id: int):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    user_bonds = [bond for bond in bonds.values() if bond["issuer"] == user_id]
    return user_bonds

@app.get("/bonds", response_model=List[Bond])
def get_bonds():
    return list(bonds.values())

# Run the server using the command: uvicorn main:app --reload
