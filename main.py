from fastapi.exceptions import RequestValidationError
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import datetime
import secrets

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

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

# In-memory databases
users = {}
tokens = {}
bonds = {}
projects = {}
current_user_id = None
block_offset = 0

dummy_user = {"id": "0x0000000000000000000000000000000000000000", "name": "Sam", "balances": {}}
users[dummy_user['id']] = dummy_user
current_user_id = dummy_user['id']

class LauncherBondContract:
    def __init__(self):
        self.bonds = {}
        self.multi_sig_owners = set()
        self.multi_sig_threshold = 2  # Example threshold
        self.pending_approvals = {}

    def add_multi_sig_owner(self, owner):
        self.multi_sig_owners.add(owner)

    def remove_multi_sig_owner(self, owner):
        self.multi_sig_owners.discard(owner)

    def create_bond_contract(self, name, symbol, total_supply, issuer, interest_rate, maturity_date, payment_schedule, is_callable=False, is_convertible=False, penalty_rate=0, requires_full_sale=True, early_withdrawal=False, adjustment_details=None):
        bond_id = generate_ethereum_address()
        bond_contract = BondContract(
            bond_id,
            name,
            symbol,
            total_supply,
            issuer,
            interest_rate,
            maturity_date,
            payment_schedule,
            is_callable,
            is_convertible,
            penalty_rate,
            requires_full_sale,
            early_withdrawal,
            adjustment_details
        )
        self.bonds[bond_id] = bond_contract
        return bond_id, bond_contract

    def halt_withdrawals(self, bond_id, owner):
        if owner not in self.multi_sig_owners:
            raise PermissionError("Only multi-sig owners can halt withdrawals.")
        if bond_id not in self.pending_approvals:
            self.pending_approvals[bond_id] = set()
        self.pending_approvals[bond_id].add(owner)
        if len(self.pending_approvals[bond_id]) >= self.multi_sig_threshold:
            bond = self.bonds[bond_id]
            bond.withdrawals_halted = True
            return True
        return False

    def stop_bond_issuance(self, owner):
        if owner not in self.multi_sig_owners:
            raise PermissionError("Only multi-sig owners can stop bond issuance.")
        self.bond_issuance_stopped = True

#Launcher Contract Instance
launcher_contract = LauncherBondContract()
launcher_contract.add_multi_sig_owner(dummy_user['id'])

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

class ERC20Snapshot(ERC20Token):
    def __init__(self, id, name, symbol, total_supply):
        super().__init__(id, name, symbol, total_supply)
        self.snapshots = []
        self.snapshot_balances = {}

    def _snapshot(self):
        current_block = get_current_block()
        self.snapshots.append(current_block)
        self.snapshot_balances[current_block] = self.balances.copy()
        return current_block

    def balance_of_at(self, account, snapshot_block):
        if snapshot_block in self.snapshot_balances:
            return self.snapshot_balances[snapshot_block].get(account, 0)
        return 0

class BondContract(ERC20Snapshot):
    def __init__(self, bond_id, name, symbol, total_supply, issuer, interest_rate, maturity_date, payment_schedule, payment_token, token_price, is_callable=False, is_convertible=False, penalty_rate=0, requires_full_sale=True, early_withdrawal=False, adjustment_details=None, min_price=0):
        super().__init__(bond_id, name, symbol, total_supply)
        self.bond_id = bond_id
        self.issuer = issuer
        self.interest_rate = interest_rate
        self.maturity_date = maturity_date
        self.payment_schedule = payment_schedule
        self.payment_token = payment_token  # Token used for payment
        self.token_price = token_price  # Price of the token
        self.is_callable = is_callable
        self.is_convertible = is_convertible
        self.penalty_rate = penalty_rate
        self.requires_full_sale = requires_full_sale
        self.early_withdrawal = early_withdrawal
        self.snapshots = []
        self.payments = []
        self.balances[issuer] = total_supply  # Issuer holds all bonds initially
        self.auction_price = total_supply  # Initialize auction price as total supply
        self.start_block = None
        self.end_block = None
        self.erc20_token = None  # Token used for payment (e.g., FRAX)
        self.bond_sold = False
        self.auction_active = False
        self.approved_payees = set([self.issuer])
        self.reentrancy_guard = False
        self.withdrawals_halted = False
        self.auto_auction_enabled = False
        self.adjustment_details = adjustment_details
        self.min_price = min_price
        self.last_adjustment_block = None
        self.total_bonded = 0  # Initial total bonded amount
        self.total_allocated = 0  # Total funds allocated for claims
        self.total_claimed = 0  # Total funds physically claimed by users
        self.snapshot_bounty = {}
        self.last_fully_paid_index = -1  # Index of the last fully paid payment


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
    
    def update_total_bonded(self, amount):
        self.total_bonded += amount  # Update the total bonded amount whenever tokens are issued
        
    def calculate_payment_due(self, snapshot_block):
        # Calculate both principal and interest
        total_payment_due = self.total_bonded * (1 + self.interest_rate / 100)
        return total_payment_due, snapshot_block

    def create_snapshot(self, caller):
        snapshot_block = self._snapshot()
        payment_due, snapshot_block = self.calculate_payment_due(snapshot_block)

        penalty = 0
        if self.total_deposited < payment_due:
            penalty = self.penalty_rate

        self.payments.append({
            "snapshot_block": snapshot_block,
            "total_amount": payment_due,
            "total_claimed": 0,
            "balances": {},  # Will be populated on user withdrawal
            "penalty": penalty,
            "snapshot_caller": caller
        })
        self.snapshot_bounty[snapshot_block] = caller
        return snapshot_block
    
    def get_owed_payments(self, holder):
        owed_amounts = []

        for payment in self.payments:
            snapshot_block = payment["snapshot_block"]
            if snapshot_block not in self.snapshots:
                continue

            snapshot = self.snapshots[snapshot_block]
            if holder in snapshot:
                holder_balance = snapshot[holder]
                total_snapshot_balance = self.total_bonded  # Use tracked total bonded amount
                allocated_amount = payment["total_allocated"]
                holder_share = (holder_balance / total_snapshot_balance) * allocated_amount
                owed_amounts.append({
                    "snapshot_block": snapshot_block,
                    "amount_owed": holder_share
                })

        return owed_amounts
    
    def deposit_payment(self, amount):
        # Access the token instance using the token address
        payment_token_instance = tokens[self.payment_token]

        # Ensure the issuer has approved enough tokens for the contract to transfer
        if payment_token_instance.allowance(self.issuer, self.bond_id) < amount:
            raise RuntimeError("Insufficient token approval.")
        
        # Transfer the tokens from issuer to contract
        if not payment_token_instance.transferFrom(self.issuer, self.bond_id, amount):
            raise RuntimeError("Transfer failed.")

        self.total_deposited += amount
        self.allocate_payments()

    def allocate_payments(self):
        remaining_amount = self.total_deposited - self.total_allocated

        # Start from the last fully paid index + 1
        for i in range(self.last_fully_paid_index + 1, len(self.payments)):
            payment = self.payments[i]
            if remaining_amount <= 0:
                break

            amount_due_with_penalty = payment["total_amount"] + (payment["total_amount"] * payment["penalty"])
            amount_due = min(amount_due_with_penalty, remaining_amount)
            remaining_amount -= amount_due
            payment["total_allocated"] = amount_due  # Track allocated funds for this snapshot
            self.total_allocated += amount_due

            # Update last fully paid index if payment is fully allocated
            if amount_due >= amount_due_with_penalty:
                self.last_fully_paid_index = i

        self.total_allocated = self.total_deposited - remaining_amount


    def claim_payment(self, holder):
        total_claimed = 0

        for payment in self.payments:
            snapshot_block = payment["snapshot_block"]
            if snapshot_block not in self.snapshots:
                continue

            snapshot = self.snapshots[snapshot_block]
            if holder in snapshot:
                holder_balance = snapshot[holder]
                total_snapshot_balance = self.total_bonded  # Use tracked total bonded amount
                allocated_amount = payment["total_allocated"]
                holder_share = (holder_balance / total_snapshot_balance) * allocated_amount
                if self.payment_token.balanceOf(self.bond_id) >= holder_share:
                    self.payment_token.transfer(self.bond_id, holder, holder_share)
                    total_claimed += holder_share
                    payment["total_allocated"] -= holder_share  # Update allocated amount

        self.total_claimed += total_claimed  # Update total claimed amount
        return total_claimed


    @only_issuer
    @reentrancy_protection
    def start_auction(self, issuer, initial_price, start_block, duration, payment_token):
        if not self.auction_active and self.start_block is None:
            self.auction_price = initial_price
            self.start_block = start_block
            self.end_block = start_block + duration if duration > 0 else None
            self.erc20_token = payment_token
            self.auction_active = True
            self.last_adjustment_block = start_block

    @reentrancy_protection
    def purchase_bond(self, buyer, payment_token_amount, bond_token_amount):
        self.apply_automatic_adjustment()
        current_block = get_current_block()

        if not self.auction_active or (self.start_block is not None and current_block < self.start_block):
            raise RuntimeError("Auction is not live.")

        expected_bond_token_amount = payment_token_amount // self.auction_price
        if bond_token_amount != expected_bond_token_amount:
            raise ValueError("The number of bond tokens does not match the payment amount submitted at the current price.")
        
        if self.balances[self.issuer] < bond_token_amount:
            raise RuntimeError("Insufficient bond inventory.")

        total_cost = bond_token_amount * self.auction_price
        
        if self.erc20_token.allowances.get(buyer, {}).get(self.bond_id, 0) < total_cost:
            raise RuntimeError("Insufficient token approval.")
        
        if not self.erc20_token.transferFrom(buyer, self.bond_id, total_cost):
            raise RuntimeError("Transfer failed.")
        
        self.transfer(self.issuer, buyer, bond_token_amount)
        
        if self.balances[self.issuer] == 0:
            self.bond_sold = True


    @only_issuer
    @reentrancy_protection
    def end_auction(self, issuer):
        current_block = get_current_block()
        if self.end_block is None or current_block >= self.end_block or (self.requires_full_sale and self.bond_sold):
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

    # Automatic auction functions
    @only_issuer
    def enable_auto_auction(self, issuer, rate):
        self.auto_auction_enabled = True
        self.auto_auction_rate = rate

    @only_issuer
    def disable_auto_auction(self, issuer):
        self.auto_auction_enabled = False

    def apply_automatic_adjustment(self):
        if self.auto_auction_enabled and self.adjustment_details:
            current_block = get_current_block()
            elapsed_blocks = current_block - self.last_adjustment_block
            adjustment_interval_blocks = (self.adjustment_details["intervalDays"] * 5760) + (self.adjustment_details["intervalHours"] * 240)

            while elapsed_blocks >= adjustment_interval_blocks:
                self.auction_price = max(self.auction_price - self.adjustment_details["amount"], self.min_price)
                self.last_adjustment_block += adjustment_interval_blocks
                elapsed_blocks -= adjustment_interval_blocks

    def get_current_price(self):
        self.apply_automatic_adjustment()
        return self.auction_price
    
    @only_issuer
    def set_min_price(self, issuer, min_price):
        self.min_price = min_price

    @only_issuer
    def set_auction_price(self, issuer, new_price):
        self.auction_price = new_price

# Pydantic models for API requests and responses
class User(BaseModel):
    id: str
    name: str
    balances: Dict[str, float] = {}

class Token(BaseModel):
    id: str
    name: str
    symbol: str
    total_supply: float
    balances: Dict[str, float] = {}

class PaymentScheduleItem(BaseModel):
    days: int
    months: int
    years: int
    amount: float

class Bond(BaseModel):
    id: str
    name: str
    symbol: str
    total_supply: float
    issuer: str
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
    auction_initial_price: float
    auction_duration: int
    auto_auction_rate: Optional[float] = None
    adjustment_details: Optional[Dict[str, int]] = None
    min_price: float = 0


class CreateUserRequest(BaseModel):
    name: str

class CreateTokenRequest(BaseModel):
    name: str
    symbol: str
    total_supply: float

class AddFundsRequest(BaseModel):
    user_id: str
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
    draft_id: Optional[str] = None
    project_info: Dict[str, str]
    bond_details: BondDetails
    auction_schedule: AuctionSchedule
    payment_schedule: List[PaymentScheduleItem]

class UserDetail(User):
    bonds_created: List[Bond] = []
    bonds_purchased: List[Bond] = []
    tokens_held: List[Token] = []
    draft_bonds: List[Dict] = []

class SetPriceRequest(BaseModel):
    price: float

class PurchaseRequest(BaseModel):
    payment_token_amount: float
    bond_token_amount: float

# Helper Conversion Functions
def convert_to_integer(amount: float, multiplier: int = 10000) -> int:
    return int(amount * multiplier)

def convert_to_blocks(date: datetime.date) -> int:
    # Assume each day is 5760 blocks (15 seconds per block)
    return (date - datetime.date(1970, 1, 1)).days * 5760

def convert_payment_schedule(payment_schedule: List[PaymentScheduleItem]) -> List[Dict[str, int]]:
    return [
        {
            "days": item.days,
            "months": item.months,
            "years": item.years,
            "amount": convert_to_integer(item.amount)
        }
        for item in payment_schedule
    ]

def get_current_block() -> int:
    # Assuming a block time of 15 seconds
    genesis_block = datetime.datetime(1970, 1, 1)
    current_time = datetime.datetime.now()
    elapsed_seconds = (current_time - genesis_block).total_seconds()
    current_block = int(elapsed_seconds // 15) + block_offset
    return current_block

def generate_ethereum_address() -> str:
    return '0x' + secrets.token_hex(20)

# API Endpoints
@app.post("/users", response_model=User)
def create_user(user: CreateUserRequest):
    user_id = generate_ethereum_address()
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
def get_user(user_id: str):
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
def switch_user(user_id: str):
    global current_user_id
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    current_user_id = user_id
    return {"message": f"Switched to user {users[user_id]['name']}"}

@app.post("/tokens", response_model=Token)
def create_token(token: CreateTokenRequest):
    token_id = generate_ethereum_address()
    tokens[token_id] = ERC20Token(token_id, token.name, token.symbol, token.total_supply)
    tokens[token_id].balances[token_id] = token.total_supply
    return {
        "id": token_id,
        "name": token.name,
        "symbol": token.symbol,
        "total_supply": token.total_supply,
        "balances": tokens[token_id].balances
    }

@app.get("/tokens", response_model=List[Token])
def get_tokens():
    return list(tokens.values())

@app.post("/users/{user_id}/issue_bond", response_model=Bond)
def issue_bond(user_id: str, bond_request: IssueBondRequest):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if launcher_contract.bond_issuance_stopped:
        raise HTTPException(status_code=403, detail="Bond issuance has been halted")

    live_block = convert_to_blocks(bond_request.maturity_date) if bond_request.maturity_date else None

    adjustment_details = {
        "intervalDays": bond_request.adjustment_details["intervalDays"],
        "intervalHours": bond_request.adjustment_details["intervalHours"],
        "amount": convert_to_integer(bond_request.adjustment_details["amount"]),
        "rate": convert_to_integer(bond_request.adjustment_details["rate"])
    } if bond_request.adjustment_details else None

    bond_id, bond_contract = launcher_contract.create_bond_contract(
        bond_request.name,
        bond_request.symbol,
        convert_to_integer(bond_request.total_supply),
        user_id,  # The user is the issuer
        convert_to_integer(bond_request.interest_rate),
        live_block,
        convert_payment_schedule(bond_request.payment_schedule),
        bond_request.is_callable,
        bond_request.is_convertible,
        convert_to_integer(bond_request.penalty_rate),
        bond_request.requires_full_sale,
        bond_request.early_withdrawal,
        adjustment_details,
        convert_to_integer(bond_request.min_price)
    )

    bond_contract.start_auction(
        user_id,
        convert_to_integer(bond_request.auction_initial_price),
        get_current_block(),
        bond_request.auction_duration,
        bond_contract.erc20_token
    )

    if bond_request.auto_auction_rate is not None:
        bond_contract.enable_auto_auction(user_id, convert_to_integer(bond_request.auto_auction_rate))

    return {"bond_id": bond_id}

@app.post("/multi_sig/halt_withdrawals/{bond_id}")
def halt_withdrawals(bond_id: str, user_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")
    
    halted = launcher_contract.halt_withdrawals(bond_id, user_id)
    if halted:
        return {"message": "Withdrawals halted successfully"}
    return {"message": "Halt withdrawals approval recorded"}

@app.post("/multi_sig/stop_bond_issuance")
def stop_bond_issuance(user_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    launcher_contract.stop_bond_issuance(user_id)
    return {"message": "Bond issuance stopped"}

@app.post("/users/{user_id}/save_draft", response_model=Dict)
def save_draft(user_id: str, draft_request: SaveDraftRequest):
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
        draft_id = generate_ethereum_address()
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
def delete_draft(user_id: str, draft_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if draft_id not in projects or projects[draft_id]["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Draft not found or does not belong to the user")

    del projects[draft_id]
    return {"message": "Draft deleted successfully"}

@app.post("/users/{user_id}/start_auction/{bond_id}")
def start_auction(user_id: str, bond_id: str, initial_price: float, duration: int):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract = launcher_contract.bonds[bond_id]
    if bond_contract.issuer != user_id:
        raise HTTPException(status_code=403, detail="Only the issuer can start the auction")

    bond_contract.start_auction(user_id, convert_to_integer(initial_price), get_current_block(), duration, bond_contract.erc20_token)
    return {"message": "Auction started"}

@app.post("/users/{user_id}/purchase_bond/{bond_id}")
def purchase_bond(user_id: str, bond_id: str, request: PurchaseRequest):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")
    
    bond_contract = launcher_contract.bonds[bond_id]
    try:
        bond_contract.purchase_bond(user_id, convert_to_integer(request.payment_token_amount), convert_to_integer(request.bond_token_amount))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Tokens purchased successfully"}

@app.post("/users/{user_id}/end_auction/{bond_id}")
def end_auction(user_id: str, bond_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract = launcher_contract.bonds[bond_id]
    if bond_contract.issuer != user_id:
        raise HTTPException(status_code=403, detail="Only the issuer can end the auction")

    if bond_contract.end_auction(user_id):
        return {"message": "Auction ended"}
    return {"message": "Auction not yet ended"}

@app.post("/users/{user_id}/withdraw_funds/{bond_id}")
def withdraw_funds(user_id: str, bond_id: str, amount: float):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract = launcher_contract.bonds[bond_id]
    if bond_contract.issuer != user_id:
        raise HTTPException(status_code=403, detail="Only the issuer can withdraw funds")

    if not bond_contract.withdraw_funds(user_id, convert_to_integer(amount)):
        raise HTTPException(status_code=400, detail="Withdrawal failed")
    return {"message": "Funds withdrawn successfully"}

@app.post("/users/{user_id}/allocate_payment/{bond_id}")
def allocate_payment(user_id: str, bond_id: str, amount: float, payment_time: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract = launcher_contract.bonds[bond_id]
    bond_contract.allocate_payment(convert_to_integer(amount), payment_time)
    return {"message": "Payment allocated successfully"}

@app.post("/bonds/{bond_id}/claim_payment/{holder}")
def claim_payment(bond_id: str, holder: str):
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract = launcher_contract.bonds[bond_id]
    try:
        total_claimed = bond_contract.claim_payment(holder)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": f"Payment of {total_claimed} claimed successfully"}


@app.post("/users/{user_id}/simulate_time_passage", response_model=Dict)
def simulate_time_passage(days: int):
    global block_offset
    block_offset += days * 5760  # Assuming 1 day = 5760 blocks (15 seconds per block)
    return {"message": "Time passage simulated successfully", "new_block_offset": block_offset}

@app.get("/users/{user_id}/bonds", response_model=List[Bond])
def get_user_bonds(user_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    user_bonds = [bond for bond in bonds.values() if bond["issuer"] == user_id]
    return user_bonds

@app.get("/bonds", response_model=List[Bond])
def get_bonds():
    return list(bonds.values())

@app.get("/current_block", response_model=Dict)
def get_current_block_info():
    current_block = get_current_block()
    return {"current_block": current_block}

# Read functions for bond status
@app.get("/bonds/{bond_id}/current_price", response_model=Dict)
def get_current_price(bond_id: str):
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract = launcher_contract.bonds[bond_id]
    current_price = bond_contract.get_current_price()
    return {"current_price": current_price}

@app.get("/bonds/{bond_id}/auction_status", response_model=Dict)
def get_auction_status(bond_id: str):
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract = launcher_contract.bonds[bond_id]
    status = {
        "auction_active": bond_contract.auction_active,
        "start_block": bond_contract.start_block,
        "end_block": bond_contract.end_block,
        "current_price": bond_contract.get_current_price()
    }
    return status

@app.post("/users/{user_id}/set_min_price/{bond_id}")
def set_min_price(user_id: str, bond_id: str, request: SetPriceRequest):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract = launcher_contract.bonds[bond_id]
    if bond_contract.issuer != user_id:
        raise HTTPException(status_code=403, detail="Only the issuer can set the minimum price")

    bond_contract.set_min_price(user_id, convert_to_integer(request.price))
    return {"message": "Minimum price set successfully"}

@app.post("/users/{user_id}/set_auction_price/{bond_id}")
def set_auction_price(user_id: str, bond_id: str, request: SetPriceRequest):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract = launcher_contract.bonds[bond_id]
    if bond_contract.issuer != user_id:
        raise HTTPException(status_code=403, detail="Only the issuer can set the auction price")

    bond_contract.set_auction_price(user_id, convert_to_integer(request.price))
    return {"message": "Auction price set successfully"}

@app.post("/bonds/{bond_id}/create_snapshot")
def create_snapshot(bond_id: str, caller: str):
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract = launcher_contract.bonds[bond_id]
    try:
        snapshot_block = bond_contract.create_snapshot(caller)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Snapshot created successfully", "snapshot_block": snapshot_block}

@app.post("/bonds/{bond_id}/deposit_payment")
def deposit_payment(bond_id: str, amount: float):
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract = launcher_contract.bonds[bond_id]
    try:
        bond_contract.deposit_payment(convert_to_integer(amount))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Payment deposited successfully"}

@app.get("/bonds/{bond_id}/owed_payments/{holder}")
def get_owed_payments(bond_id: str, holder: str):
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract = launcher_contract.bonds[bond_id]
    owed_payments = bond_contract.get_owed_payments(holder)

    return {"owed_payments": owed_payments}

# Run the server using the command: uvicorn main:app --reload
