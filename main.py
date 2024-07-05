from fastapi.exceptions import RequestValidationError
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import datetime
import secrets
from dataclasses import dataclass, asdict
from enum import Enum

BLOCKS_PER_DAY = 24 * 60 * 4 # Hours * minutes * 4 = 15 seconds per block

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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    logger.info(f"Request headers: {request.headers}")
    logger.info(f"Request query params: {request.query_params}")
    body = await request.body()
    logger.info(f"Request body: {body}")
    response = await call_next(request)
    return response

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
frax_token_id = ""

dummy_user = {"id": "0x0000000000000000000000000000000000000000", "name": "Sam", "balances": {}}
users[dummy_user['id']] = dummy_user
current_user_id = dummy_user['id']

# Simulating Structs for passing data to the contract using dataclasses
@dataclass
class AuctionDurationStruct:
    days: int
    hours: int

@dataclass
class AdjustmentDetailsStruct:
    intervalDays: int
    intervalHours: int
    amount: int
    rate: int

@dataclass
class BondDurationStruct:
    years: int
    months: int
    days: int

@dataclass
class FixedPaymentIntervalStruct:
    days: int
    months: int
    years: int

@dataclass
class CustomRepaymentIntervalStruct:
    days: int
    months: int
    years: int
    principalPercent: int
    interestPercent: int

@dataclass
class DateStruct:
    hours: int
    days: int
    months: int
    years: int

@dataclass
class PaymentStruct:
    snapshot_block: int
    total_amount: int
    total_allocated: int
    total_claimed: int
    balances: Dict[str, int]
    penalty_paid: int
    snapshot_caller: str

@dataclass
class ProjectInfoStruct:
    name: str
    description: str
    website: str
    imageUrl: str
    coinGeckoUrl: str

@dataclass
class AuctionScheduleStruct:
    auctionType: str
    auctionDuration: AuctionDurationStruct
    auctionEndCondition: str
    adjustAutomatically: bool
    adjustmentType: str
    adjustmentDetails: AdjustmentDetailsStruct
    minPrice: int
    startAutomatically: bool
    startBlock: int
    endBlock: int

@dataclass
class BondDetailsStruct:
    title: str
    totalAmount: int
    infiniteTokens: bool
    tokens: int
    tokenPrice: int
    tokenSymbol: str
    interestRate: int
    requiresFullSale: bool
    earlyRepayment: bool
    collateral: bool
    paymentTokenAddress: str

@dataclass
class BondRepaymentStruct:
    bondDuration: BondDurationStruct
    bondTotalDurationBlocks: int
    repaymentType: str
    paymentSchedule: str
    latePenalty: int
    fixedPaymentInterval: FixedPaymentIntervalStruct
    customRepaymentSchedule: List[CustomRepaymentIntervalStruct]

@dataclass
class SaveDraftRequestStruct:
    project_info: ProjectInfoStruct
    bond_details: BondDetailsStruct
    auction_schedule: AuctionScheduleStruct
    bond_repayment: BondRepaymentStruct
    draft_id: Optional[str] = None


class ApproveRequest(BaseModel):
    user_id: str
    spender: str
    amount: float

class SwapRequest(BaseModel):
    sendTokenAddress: str
    receiveTokenAddress: str
    sendAmount: float
    receiveAmount: float
    user_id: str

class SwapResponse(BaseModel):
    balances: Dict[str, float]

class CreateSnapshotRequest(BaseModel):
    bond_id: str
    user_id: str

class BondState(Enum):
    AUCTION_NOT_STARTED = 0
    AUCTION_LIVE = 1
    BOND_LIVE = 2
    ACTIVITY_HALTED = 3
    BOND_CANCELLED = 4
    BOND_ENDED = 5
    BOND_ERROR = 6

class BlockChainState:
    def __init__(self):
        genesis_block = datetime.datetime(1970, 1, 1)
        current_time = datetime.datetime.now()
        elapsed_seconds = (current_time - genesis_block).total_seconds()
        self.current_block = int(elapsed_seconds // 15) + block_offset

    def advance_blocks(self, amount_to_advance):
        self.current_block += amount_to_advance

    def set_current_block(self, block_to_set):
        self.current_block = block_to_set

    def get_current_block(self):
        return self.current_block

class LauncherBondContract:
    def __init__(self):
        self.bonds = {}
        self.issuer_bonds = {}
        self.multi_sig_owners = set()
        self.multi_sig_threshold = 2  # Example threshold
        self.pending_approvals = {}
        self.bond_issuance_stopped = False


    def add_multi_sig_owner(self, owner):
        self.multi_sig_owners.add(owner)

    def remove_multi_sig_owner(self, owner):
        self.multi_sig_owners.discard(owner)

    def create_bond_contract(self, issuer, project_info: ProjectInfoStruct, bond_details: BondDetailsStruct, auction_schedule: AuctionScheduleStruct, bond_repayment: BondRepaymentStruct):
        bond_contract = BondContract(
            issuer,
            project_info,
            bond_details,
            auction_schedule,
            bond_repayment
        )
        self.bonds[bond_contract.contract_address] = bond_contract
        issuer_bonds = self.issuer_bonds.get(issuer, [])
        if issuer not in self.issuer_bonds:
            self.issuer_bonds[issuer] = [bond_contract.contract_address]
        else:
            self.issuer_bonds[issuer].append(bond_contract.contract_address)
        return bond_contract.contract_address, bond_contract

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

# Blockchain State Instance
blockchain_state = BlockChainState()
# Classes for ERC20Token and Bond operations
class ERC20Token:
    def __init__(self, name, symbol, total_supply):
        self.contract_address = generate_ethereum_address()
        self.name = name
        self.symbol = symbol
        self.total_supply = total_supply *10**18
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
    
    def allowanceOf(self, owner, spender):
        return self.allowances.get(owner, {}).get(spender, 0)

    def transferFrom(self, spender, owner, receiver, amount):
        if self.allowanceOf(owner, spender) >= amount and self.balanceOf(owner) >= amount:
            self.allowances[owner][spender] -= amount
            self.balances[owner] -= amount
            self.balances[receiver] = self.balances.get(receiver, 0) + amount
            return True
        return False

class ERC20Snapshot(ERC20Token):
    def __init__(self, name, symbol, total_supply):
        super().__init__(name, symbol, total_supply)
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
    def __init__(self, issuer, project_info: ProjectInfoStruct, bond_details: BondDetailsStruct, auction_schedule: AuctionScheduleStruct, bond_repayment: BondRepaymentStruct):
        super().__init__(bond_details.title, bond_details.tokenSymbol, 0)
        self.issuer = issuer
        self.project_info: ProjectInfoStruct = project_info
        self.bond_details: BondDetailsStruct = bond_details
        self.auction_schedule: AuctionScheduleStruct = auction_schedule
        self.bond_repayment: BondRepaymentStruct = bond_repayment
        self.activity_halted = False
        
        # Bond state variables
        self.auction_start_block = auction_schedule.startBlock
        self.auction_end_block = auction_schedule.endBlock
        self.bond_end_block = 0
        self.bond_manually_cancelled = False

        self.payments = []
        self.payments_index : Dict[str: int] = {}

        self.total_repaid = 0
        self.last_fully_paid_index = -1

        self.snapshot_bounty : Dict[int, str] = {}
        self.overpayment = 0
        self.withdrawn_funds = False
        self.funds_from_bond_purchase = 0

        self.approved_payees = {}

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
    
    def convert_date_to_blocks(self, hours=0, days = 0, months = 0, years = 0):
        # Assume each day is 5760 blocks (15 seconds per block)
        return (years * 365 * 5760) + (months * 30 * 5760) + (days * 5760) + (hours * 240)

    def convert_blocks_to_date(self, blocks):
        return DateStruct(hours=blocks // 240, days=blocks // 5760, months=blocks // (5760 * 30), years=blocks // (5760 * 365))
    
    def elapsed_blocks_since_last_snapshot(self):
        if len(self.snapshots) == 0:
            return 0
        if len(self.snapshots) == 1:
            return get_current_block() - self.snapshots[0]
        return get_current_block() - self.snapshots[-1]
                    
    def _calculate_snapshot_payment_due(self, snapshot_block):
        if snapshot_block not in self.snapshot_balances:
            return 0
        start_block = self.auction_end_block
        if len(self.snapshots) > 1:
            start_block = self.snapshots[-2]
        end_snapshot = self.auction_end_block + self.bond_repayment.bondTotalDurationBlocks
        snapshot_block = min(snapshot_block, end_snapshot)
        if start_block >= snapshot_block:
            return 0
        total_amount = (self.total_supply * self.bond_details.tokenPrice) / 10**18
        if self.bond_repayment.paymentSchedule == "fixed":
            interest_due = self._get_apr_amount_for_block_window(
                total_amount=total_amount,
                interest_rate=self.bond_details.interestRate,
                start_block=start_block,
                end_block=snapshot_block
            )
            principal_due = 0
            if self.bond_repayment.repaymentType == "interest-only" and snapshot_block == end_snapshot:
                # If it's an interest-only bond, the principal is due at the end
                principal_due = total_amount
            if self.bond_repayment.repaymentType == "principal-interest":
                elapsed_blocks = snapshot_block - start_block
                principal_per_block = total_amount // (self.bond_repayment.bondTotalDurationBlocks)
                principal_due = elapsed_blocks * principal_per_block
            return principal_due + interest_due
        else:
            raise NotImplementedError("Custom repayment schedules not yet supported.")
    
    def next_eligible_snapshot(self):
        status = self.get_bond_status()
        if status != BondState.BOND_LIVE:
            return 0
        
        last_snapshot_block = 0
        if len(self.snapshots) == 0:
            last_snapshot_block = self.auction_end_block
        else:
            last_snapshot_block = self.snapshots[-1]
        bond_end_block =  self.bond_repayment.bondTotalDurationBlocks+self.auction_end_block
        if self.bond_repayment.paymentSchedule == "fixed":
            days = self.bond_repayment.fixedPaymentInterval.days
            months = self.bond_repayment.fixedPaymentInterval.months
            years = self.bond_repayment.fixedPaymentInterval.years
            return min(last_snapshot_block + self.convert_date_to_blocks(days=days, months=months, years=years), bond_end_block)
        else:
            custom_repayment_index = len(self.snapshots)
            if custom_repayment_index < len(self.bond_repayment.customRepaymentSchedule):
                days = self.bond_repayment.customRepaymentSchedule[custom_repayment_index].days
                months = self.bond_repayment.customRepaymentSchedule[custom_repayment_index].months
                years = self.bond_repayment.customRepaymentSchedule[custom_repayment_index].years
                return min(last_snapshot_block + self.convert_date_to_blocks(days=days, months=months, years=years), bond_end_block)
            else:
                return max(last_snapshot_block, bond_end_block)

    def create_snapshot(self, caller):
        status = self.get_bond_status()
        if status != BondState.BOND_LIVE:
            raise RuntimeError("Bond is not live.")
        next_snapshot_block = self.next_eligible_snapshot()
        curr_block = get_current_block()
        if curr_block < next_snapshot_block:
            raise RuntimeError("Not yet eligible for snapshot.")
        snapshot_block = self._snapshot()
        payment_due = self._calculate_snapshot_payment_due(snapshot_block)
        self._add_payment(
            PaymentStruct(
                snapshot_block=snapshot_block, 
                total_amount=payment_due, 
                total_allocated=0, 
                total_claimed=0,
                balances={}, 
                penalty_paid=0,
                snapshot_caller=caller
            )
        )
        self.snapshot_bounty[snapshot_block] = caller
        return snapshot_block
    
    @only_issuer
    def cancel_bond(self, issuer, is_cancelled = True):
        # TODO: Need to actually set logic for checking whether the bond is eligible to be cancelled
        self.bond_manually_cancelled = is_cancelled
    
    def _add_payment(self, payment):
        self.payments.append(payment)
        self.payments_index[payment.snapshot_block] = len(self.payments) - 1
    
    def _get_payment_for_block(self, snapshot_block):
        if snapshot_block not in self.payments_index:
            raise RuntimeError("Payment not found.")
        return self.payments[self.payments_index[snapshot_block]]
    
    def _get_apr_per_block(self, total_amount, interest_rate):
        apr_amount = (interest_rate * total_amount) // 10**20
        return apr_amount // (5760 * 365)
    
    def _get_apr_amount_for_block_window(self, total_amount, interest_rate, start_block, end_block):
        apr_per_block = self._get_apr_per_block(total_amount, interest_rate)
        if start_block >= end_block:
            raise ValueError("Start block must be less than end block.")
        elapsed_blocks = end_block - start_block
        return elapsed_blocks * apr_per_block
    
    def get_amount_owed_for_snapshot(self, snapshot_block):
        if snapshot_block not in self.snapshot_balances:
            return [0,0]
        curr_block = get_current_block()

        payment_info: PaymentStruct = self._get_payment_for_block(snapshot_block)
        amount_due = payment_info.total_amount - payment_info.total_allocated
        if amount_due <= 0:
            return [0, 0]
        payment_due_block = snapshot_block + 5760 * 1
        penalty_amount_remaining = 0
        if curr_block > payment_due_block and self.bond_repayment.latePenalty > 0:
            # The issuer has 7 days to allocate funds to the snapshot
            # If they don't allocate funds within this time, the penalty is applied
            penalty_amount = self._get_apr_amount_for_block_window(
                total_amount= amount_due,
                interest_rate=self.bond_repayment.latePenalty,
                start_block=payment_due_block,
                end_block=curr_block
            )
            penalty_amount_remaining = penalty_amount - payment_info.penalty_paid
        return (amount_due, penalty_amount_remaining)
    
    def get_total_owed(self):
        total_principal_owed = 0
        total_penalty_owed = 0
        for i in range(self.last_fully_paid_index + 1, len(self.payments)):
            payment: PaymentStruct = self.payments[i]
            principal, penalty = self.get_amount_owed_for_snapshot(payment.snapshot_block)
            total_principal_owed += principal
            total_penalty_owed += penalty
        return (total_principal_owed, total_penalty_owed)
    
    def get_total_owed_breakdown(self):
        total_owed = []
        for i in range(self.last_fully_paid_index + 1, len(self.payments)):
            payment: PaymentStruct = self.payments[i]
            principal, penalty = self.get_amount_owed_for_snapshot(payment.snapshot_block)
            total_owed += [[payment.snapshot_block, principal, penalty]]
        return total_owed
    
    @reentrancy_protection
    def deposit_payment(self, amount):
        status = self.get_bond_status()
        if status != BondState.BOND_LIVE:
            raise RuntimeError("Bond is not live.")
        
        # Access the token instance using the token address
        payment_token_instance: ERC20Token = tokens[self.bond_details.paymentTokenAddress]
        
        # Ensure the issuer has approved enough tokens for the contract to transfer
        if payment_token_instance.allowanceOf(self.issuer, self.contract_address) < amount:
            raise RuntimeError("Insufficient token approval.")
        
        # Transfer the tokens from issuer to contract
        # This might create an over payment situation
        # The overpayment will be allocated to the next snapshot
        # If the issuer realizes they have overpaid they can withdraw unallocated funds
        if not payment_token_instance.transferFrom(self.contract_address, self.issuer, self.contract_address, amount):
            raise RuntimeError("Transfer failed.")
        
        self._allocate_payments(amount)
        
        
    
    def _allocate_payments(self, amount):
        amount += self.overpayment
        
        repaid = 0
        
        for i in range(self.last_fully_paid_index + 1, len(self.payments)):
            payment: PaymentStruct = self.payments[i]
            principal_due, penalty_due = self.get_amount_owed_for_snapshot(payment.snapshot_block)
            penalty_payment_amount = 0
            principal_payment_amount = 0
            finished_payment = False

            if amount >= principal_due + penalty_due:
                penalty_payment_amount = penalty_due
                principal_payment_amount = principal_due
                finished_payment = True
            elif penalty_due >= amount:
                penalty_payment_amount = amount
            else:
                penalty_payment_amount = penalty_due
                principal_payment_amount = amount - penalty_due
            
            payment.total_allocated += principal_payment_amount
            payment.penalty_paid += penalty_payment_amount
            repaid += principal_payment_amount
            if finished_payment:
                self.last_fully_paid_index += 1
                finished_payment = False
            amount -= penalty_payment_amount + principal_payment_amount
            if amount <= 0:
                break
        
        if amount > 0:
            # Overpayment
            self.overpayment = amount
            
        self.total_repaid += repaid
        if self.total_repaid >= self.total_supply * self.bond_details.tokenPrice / 10**18:
            self.bond_end_block = get_current_block()
            
                
    @only_issuer
    @reentrancy_protection
    def withdraw_overpayment(self, issuer):
        if self.overpayment > 0:
            payment_token_instance = tokens[self.bond_details.paymentTokenAddress]
            if not payment_token_instance.transfer(self.bond_id, issuer, self.overpayment):
                raise RuntimeError("Transfer failed.")
            self.overpayment = 0

    def get_amount_user_entitled_to_for_snapshot(self, holder, snapshot_block):
        balance_at_snapshot = self.balance_of_at(holder, snapshot_block)
        if balance_at_snapshot == 0:
            return [0, 0, 0, 0]
        
        payment_info: PaymentStruct = self._get_payment_for_block(snapshot_block)
        principal_percentage_owed = (payment_info.total_allocated * balance_at_snapshot) // self.total_supply
        penalty_percentage_owed = (payment_info.penalty_paid * balance_at_snapshot) // self.total_supply
        amount_already_claimed = payment_info.balances.get(holder, 0)
        available_to_claim = principal_percentage_owed + penalty_percentage_owed - amount_already_claimed

        return [principal_percentage_owed, penalty_percentage_owed, amount_already_claimed, available_to_claim]
    
    def get_amount_user_entitled_to(self, holder):
        amounts = []
        for i in range(len(self.payments)):
            payment:PaymentStruct = self.payments[i]
            snapshot_block = payment.snapshot_block
            amounts += [[snapshot_block] + self.get_amount_user_entitled_to_for_snapshot(holder, snapshot_block)]
        return amounts

    @reentrancy_protection
    def claim_payment_for_snapshot(self, holder, snapshot_block):
        balance_at_snapshot = self.balance_of_at(holder, snapshot_block)
        if balance_at_snapshot == 0:
            return 0
        payment_info = self._get_payment_for_block(snapshot_block)
        principal_percentage_owed = (payment_info.total_allocated * balance_at_snapshot) // self.total_supply
        penalty_percentage_owed = (payment_info.penalty_paid * balance_at_snapshot) // self.total_supply
        amount_already_claimed = payment_info.balances.get(holder, 0)
        available_to_claim = principal_percentage_owed + penalty_percentage_owed - amount_already_claimed
        if available_to_claim == 0:
            return 0
        # Transfer token to holder
        payment_token_instance = tokens[self.bond_details.paymentTokenAddress]
        if not payment_token_instance.transfer(self.bond_id, holder, available_to_claim):
            raise RuntimeError("Transfer failed.")
        payment_info.balances[holder] += available_to_claim
        payment_info.total_claimed += available_to_claim
    
    @reentrancy_protection
    def claim_payments(self, holder):
        total_claimed = 0
        for payment in self.payments:
            payment: PaymentStruct
            balance_at_snapshot = self.balance_of_at(holder, payment.snapshot_block)
            if balance_at_snapshot == 0:
                continue
            principal_percentage_owed = (payment.total_allocated * balance_at_snapshot) // self.total_supply
            penalty_percentage_owed = (payment.penalty_paid * balance_at_snapshot) // self.total_supply
            amount_already_claimed = payment.balances.get(holder, 0)
            available_to_claim = principal_percentage_owed + penalty_percentage_owed - amount_already_claimed
            if available_to_claim == 0:
                continue
            payment.balances[holder] = payment.balances.get(holder,0) + available_to_claim
            payment.total_claimed += available_to_claim
            total_claimed += available_to_claim
        # Transfer token to holder
        payment_token_instance: ERC20Token = tokens[self.bond_details.paymentTokenAddress]
        if not payment_token_instance.transfer(self.contract_address, holder, total_claimed):
            raise RuntimeError("Transfer failed.")


    @only_issuer
    @reentrancy_protection
    def start_auction(self, issuer):
        status = self.get_bond_status()
        if status != BondState.AUCTION_NOT_STARTED:
            raise RuntimeError("Auction has already started.")
        self.auction_start_block = get_current_block()            
        
    def get_remaining_tokens(self):
        if self.bond_details.infiniteTokens:
            return 1
        return self.bond_details.tokens - self.total_supply
    
    def is_bond_sold_out(self):
        if self.bond_details.infiniteTokens:
            return False
        return self.total_supply >= self.bond_details.tokens

    @only_issuer
    def halt_activity(self, issuer, status):
        if type(status) != bool:
            raise TypeError("Status should be a boolean.")
        self.activity_halted = status
    
    def _is_pre_auction(self):
        # Only called internally
        # If we don't have a start block then the auction is manual and hasn't started yet
        curr_block = get_current_block()
        return self.auction_start_block == 0 or curr_block < self.auction_start_block
    
    def _is_auction_live(self):
        # Only called internally
        # When bond is purchased we will check to see if all tokens are sold and then set an end block if they are
        # If they have a manual end block then we will set that as the end block unless all tokens are sold before that
        curr_block = get_current_block()
        return self.auction_start_block != 0 and self.auction_start_block <= curr_block \
            and (self.auction_end_block == 0 or self.auction_end_block != 0 and curr_block <= self.auction_end_block)
    
    def _is_bond_live(self):
        # Only called internally
        # bond_end_block is set when the issuer repays the final repayment 
        # or makes all payments and they want to end early
        curr_block = get_current_block()
        return self.auction_end_block != 0 and self.auction_end_block <= curr_block \
            and self.bond_end_block == 0
    
    def _is_bond_finished(self):
        # Only called internally
        # If the bond has ended and all payments have been made
        return self.bond_end_block != 0


    def _is_bond_cancelled(self):
        # Only called internally
        # If bond auction has started and ended and not all tokens are sold but required to be
        curr_block = get_current_block()
        auction_over = self.auction_end_block != 0 and self.auction_end_block <= curr_block
        return self.bond_manually_cancelled \
            or (auction_over and self.bond_details.requiresFullSale and not self.is_bond_sold_out())

    
    def get_bond_status(self) -> BondState:
        # states:
        # 0 - auction not started
        # 1 - auction started
        # 2 - auction ended/bond is live
        # 3 - activity halted
        # 4 - bond cancelled
        current_block = get_current_block()
        passed_auction_start_block = False
        if self._is_bond_cancelled():
            return BondState.BOND_CANCELLED
        if self.activity_halted:
            return BondState.ACTIVITY_HALTED
        if self._is_pre_auction():
            return BondState.AUCTION_NOT_STARTED
        if self._is_auction_live():
            return BondState.AUCTION_LIVE
        if self._is_bond_live():
            return BondState.BOND_LIVE
        if self._is_bond_finished():    
            return BondState.BOND_ENDED
        return BondState.BOND_ERROR
        

    # @reentrancy_protection
    def purchase_bond(self, buyer, payment_token_amount, bond_token_amount):
        auction_price = self.get_current_auction_price()
        status = self.get_bond_status()
        if status != BondState.AUCTION_LIVE:
            raise RuntimeError("Auction is not live.")

        expected_bond_token_amount = payment_token_amount * 10**18 // auction_price 
        if bond_token_amount < expected_bond_token_amount:
            raise ValueError("The number of bond tokens does not match the payment amount submitted at the current price.")
        
        bond_tokens_to_transfer = min(bond_token_amount, expected_bond_token_amount)
        
        if not self.bond_details.infiniteTokens and self.bond_details.tokens - self.total_supply < bond_tokens_to_transfer:
            raise RuntimeError("Insufficient bond inventory.")

        total_cost = bond_tokens_to_transfer * auction_price // 10**18
        
        payment_token: ERC20Token = tokens[self.bond_details.paymentTokenAddress]
        allowance = payment_token.allowanceOf(buyer, self.contract_address)
        if allowance < total_cost:
            raise RuntimeError("Insufficient token approval.")
        
        if not payment_token.transferFrom(self.contract_address, buyer, self.contract_address, total_cost):
            raise RuntimeError("Transfer failed.")
        
        self.total_supply += bond_tokens_to_transfer
        self.balances[buyer] = self.balances.get(buyer,0) + bond_tokens_to_transfer
        self.funds_from_bond_purchase += total_cost

        if self.is_bond_sold_out():
            self.auction_end_block = get_current_block()


    @only_issuer
    @reentrancy_protection
    def end_auction(self, issuer):
        status = self.get_bond_status()
        if status != BondState.AUCTION_LIVE:
            raise RuntimeError("Auction is not live.")
        current_block = get_current_block()
        self.auction_end_block = current_block

    @only_issuer
    @reentrancy_protection
    def withdraw_funds(self, issuer):
        status = self.get_bond_status()
        if status != BondState.BOND_LIVE:
            raise RuntimeError("Bond is not live.")
        if self.withdrawn_funds:
            return False
        funds_token: ERC20Token = tokens[self.bond_details.paymentTokenAddress]
        if not funds_token.transfer(self.contract_address, self.issuer, self.funds_from_bond_purchase):
            raise RuntimeError("Transfer failed.")
        self.withdrawn_funds = True
        return True

    @only_issuer
    def approve_payee(self, issuer, payee):
        self.approved_payees[payee] = True
    
    @only_issuer
    def remove_payee(self, issuer, payee):
        self.approved_payees[payee] = False

    def convert_to_shares(self, holder, conversion_rate, shares_contract):
        if self.is_convertible:
            num_shares = self.balances[holder] * conversion_rate
            shares_contract.transfer(self.issuer, holder, num_shares)
            self.balances[holder] = 0

    # Automatic auction functions
    @only_issuer
    def enable_auto_auction(self, issuer, rate):
        self.auto_auction_enabled = True
        self.auto_auction_rate = rate

    @only_issuer
    def disable_auto_auction(self, issuer):
        self.auto_auction_enabled = False
    
    def get_current_auction_price(self):
        price = self.bond_details.tokenPrice
        status = self.get_bond_status()
        if status != BondState.AUCTION_LIVE:
            return price
        if self.auction_schedule.adjustAutomatically:
            curr_block = get_current_block()
            elapsed_blocks = curr_block - self.auction_start_block
            adjustment_interval_blocks = self.convert_date_to_blocks(days=self.auction_schedule.adjustmentDetails.intervalDays, hours=self.auction_schedule.adjustmentDetails.intervalHours)
            if adjustment_interval_blocks == 0:
                return price
            intervals_passed = elapsed_blocks // adjustment_interval_blocks # Make sure that we don't divide by 0 in the contract
            price_interval_delta = 0
            if self.auction_schedule.adjustmentType == "percentage":
                price_interval_delta = (price * self.auction_schedule.adjustmentDetails.rate)// 10**20
            elif self.auction_schedule.adjustmentType == "fixed":
                price_interval_delta = self.auction_schedule.adjustmentDetails.amount
            price = max(price - (price_interval_delta * intervals_passed), self.auction_schedule.minPrice)
        return price

# Pydantic models for API requests and responses
class User(BaseModel):
    id: str
    name: str
    balances: Dict[str, float] = {}

class Token(BaseModel):
    contract_address: str
    name: str
    symbol: str
    total_supply: float
    balances: Dict[str, float] = {}

class UserTokenDetails(BaseModel):
    contract_address: str
    name: str
    symbol: str
    total_supply: float
    balance: float

class PaymentScheduleItem(BaseModel):
    days: int
    months: int
    years: int
    amount: float

class BondDetailsRequest(BaseModel):
    user_id: str

class IssueBondRequest(BaseModel):
    user_id: str
    draft_id: str


class CreateUserRequest(BaseModel):
    name: str

class CreateTokenRequest(BaseModel):
    name: str
    symbol: str
    total_supply: float

class UpdateBlocksByDaysRequest(BaseModel):
    days_to_advance: str

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

class CustomRepaymentInterval(BaseModel):
    days: int
    months: int
    years: int
    principalPercent: float
    interestPercent: float

class ProjectInfo(BaseModel):
    name: str
    description: str
    website: str
    imageUrl: str
    coinGeckoUrl: str

class AuctionSchedule(BaseModel):
    auctionType: str
    auctionDuration: AuctionDuration
    auctionEndCondition: str
    adjustAutomatically: bool
    adjustmentType: str
    adjustmentDetails: AdjustmentDetails
    minPrice: float
    startAutomatically: bool
    startDate: str


    @property
    def startBlock(self):
        if self.startAutomatically and self.startDate != "":
            formats = ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"]
            for fmt in formats:
                try:
                    return convert_to_blocks(datetime.datetime.strptime(self.startDate, fmt))
                except ValueError:
                    continue
            raise ValueError(f"Date {self.startDate} does not match any known formats")
        else:
            return 0
        
    @property
    def endBlock(self):
        if self.startDate!= "" and self.auctionDuration.hours > 0 and self.auctionDuration.days > 0 and self.auctionEndCondition == "hard-end":
            return convert_to_blocks(self.start_date) + convert_date_to_blocks(hours=self.auctionDuration.hours, days=self.auctionDuration.days)
        else:
            return 0
        
    def dict(self, *args, **kwargs):
        data = super().dict(*args, **kwargs)
        data['startBlock'] = self.startBlock
        data['endBlock'] = self.endBlock
        return data

class BondDetails(BaseModel):
    title: str
    totalAmount: int
    infiniteTokens: bool
    tokens: int
    tokenPrice: float
    tokenSymbol: str
    interestRate: float
    requiresFullSale: bool
    earlyRepayment: bool
    collateral: bool
    paymentTokenAddress: Optional[str] = frax_token_id

    def __str__(self):
        return f"title: {str(self.title)}, totalAmount: {str(self.totalAmount)}, infiniteTokens: {str(self.infiniteTokens)}, tokens: {str(self.tokens)}, tokenPrice: {str(self.tokenPrice)}, tokenSymbol: {str(self.tokenSymbol)}, interestRate: {str(self.interestRate)}, requiresFullSale: {str(self.requiresFullSale)}, earlyRepayment: {str(self.earlyRepayment)}, collateral: {str(self.collateral)}, paymentTokenAddress: {str(self.paymentTokenAddress)}"

class BondRepayment(BaseModel):
    bondDuration: BondDuration
    repaymentType: str
    paymentSchedule: str
    latePenalty: float
    fixedPaymentInterval: FixedPaymentInterval
    customRepaymentSchedule: List[CustomRepaymentInterval]

class Bond(BaseModel):
    bond_id: str

class PublishedBondPreview(BaseModel):
    contract_address: str
    name : str
    status: str
    apr: float
    interest: float
    duration: BondDuration
    token_price: float
    tokens: int
    total_amount: int
    total_supply: int
    auction_start_block: int
    auction_start_date: str
    auction_end_block: int
    auction_end_date: str
    next_snap_shot_block: int
    image_url: str
    late_penalty: float
    bond_end_date: str

class SaveDraftRequest(BaseModel):
    draft_id: Optional[str] = None
    project_info: ProjectInfo
    bond_details: BondDetails
    auction_schedule: AuctionSchedule
    bond_repayment: BondRepayment

class UserDetail(User):
    bonds_created: List[PublishedBondPreview] = []
    bonds_purchased: List[PublishedBondPreview] = []
    tokens_held: List[UserTokenDetails] = []
    draft_bonds: List[Dict] = []

class SetPriceRequest(BaseModel):
    price: float

class PurchaseRequest(BaseModel):
    user_id: str
    bond_id: str
    payment_token_address: str
    payment_token_amount: float
    bond_token_amount: float

class PaymentRecord(BaseModel):
    snapshot_block: int
    total_amount: int
    total_allocated: int
    total_claimed: int
    balances: Dict[str, int]
    penalty_paid: int
    snapshot_caller: str

class AmountUserEntitledToRecord(BaseModel):
    snapshot_block: int
    principal_owed: float
    penalty_owed: float
    amount_already_claimed: float
    available_to_claim: float

class IssuerOwedBreakdown(BaseModel):
    snapshot_block: int
    principal_owed: float
    penalty_owed: float

class TotalIssuerOwes(BaseModel):
    total_principal_owed: float
    total_penalty_owed: float

class PublishedBondDetails(BaseModel):
    contract_address: str
    issuer: str
    total_supply: int
    project_info: ProjectInfo
    bond_details: BondDetails
    auction_schedule: AuctionSchedule
    bond_repayment: BondRepayment
    activity_halted: bool
    withdrawn_funds: bool
    funds_from_bond_purchase: float

    auction_start_block: int
    auction_end_block: int
    bond_end_block: int
    bond_manually_cancelled: bool
    bond_end_date: str
    auction_end_date: str

    overpayment: float

    # Computed properties
    current_block: int
    next_eligible_snapshot: int
    total_issuers_owes: TotalIssuerOwes
    total_issuer_owes_break_down: List[IssuerOwedBreakdown]
    amount_user_entitled_to_and_claimable : List[AmountUserEntitledToRecord] # This is only what they can currently claim
    remaining_tokens : float
    bond_sold_out : bool
    bond_status: str
    current_auction_price: float
    apr : float







# Helper Conversion Functions
def convert_to_integer(amount: float, multiplier: int = 10**18) -> int:
    return int(amount * multiplier)

def convert_to_float(amount: int, multiplier: int = 10**18) -> float:
    return amount / multiplier

def convert_to_blocks(date: datetime) -> int:
    # Assume each day is 5760 blocks (15 seconds per block)
    return (date.date() - datetime.date(1970, 1, 1)).days * 5760

def convert_date_to_blocks(hours=0, days = 0, months = 0, years = 0):
    # Assume each day is 5760 blocks (15 seconds per block)
    return (years * 365 * 5760) + (months * 30 * 5760) + (days * 5760) + (hours * 240)

def convert_blocks_to_date_str(block_number: int) -> datetime:
    # Assume each day is 5760 blocks (15 seconds per block)
    if block_number == 0:
        return ""
    return (datetime.datetime(1970, 1, 1) + datetime.timedelta(days=block_number // 5760)).isoformat()

def convert_days_to_blocks(days: float):
    # Assumes 15 second blocks
    return days * BLOCKS_PER_DAY

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
    return blockchain_state.get_current_block()

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
    token : ERC20Token = tokens[token_id]
    if not token.transfer(token_id, request.user_id, convert_to_integer(request.amount)):
        raise HTTPException(status_code=500, detail="Transfer failed")
    

    return {"message": "Funds added successfully", "new_balance": convert_to_float(token.balanceOf(request.user_id))}

@app.get("/users/{user_id}", response_model=UserDetail)
def get_user(user_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users[user_id]
    
    # Retrieve bonds created by the user
    user_bonds = [launcher_contract.bonds[bond_id] for bond_id in launcher_contract.issuer_bonds.get(user_id, [])]
    user_created_bond_previews = [build_bond_preview_from_bond(bond) for bond in user_bonds]
    # user["bonds_created"] = user_bonds
    user["bonds_created"] = user_created_bond_previews
    # Retrieve bonds purchased by the user
    user_bonds_purchased = [bond for bond in bonds.values() if user_id in bond["balances"] and bond["balances"][user_id] > 0]
    user["bonds_purchased"] = user_bonds_purchased
    
    # Retrieve tokens held by the user
    user_tokens = [token for token in tokens.values() if user_id in token.balances and token.balances[user_id] > 0]
    tokens_held = []
    for token in user_tokens:
        token: ERC20Token
        rec = {
            "contract_address": token.contract_address,
            "name": token.name,
            "symbol": token.symbol,
            "balance": convert_to_float(token.balances[user_id]),
            "total_supply": convert_to_float(token.total_supply)
        }
        tokens_held.append(rec)
    user["tokens_held"] = tokens_held
    
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
    token = ERC20Token(token.name, token.symbol, token.total_supply)
    tokens[token.contract_address] = token 
    tokens[token.contract_address].balances[token.contract_address] = token.total_supply
    return {
        "id": token.contract_address,
        "name": token.name,
        "symbol": token.symbol,
        "total_supply": token.total_supply,
        "balances": tokens[token.contract_address].balances
    }

frax_token_info = CreateTokenRequest(name="Frax", symbol="FRAX", total_supply=100000000)
frax_token_id = create_token(frax_token_info)['id']

@app.get("/tokens", response_model=List[Token])
def get_tokens():
    return list(tokens.values())

@app.post("/users/{user_id}/issue_bond", response_model=Bond)
def issue_bond(user_id: str, request: SaveDraftRequest):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if launcher_contract.bond_issuance_stopped:
        raise HTTPException(status_code=403, detail="Bond issuance has been halted")
    
    # TODO: auction startDate needs to be converted to block number and changed to startBlock
    # TODO: Currently auction only ends based on whether it is sold out or manually ended
    # TODO: need an auction end data and convert to block number
    # TODO: timezone is not used in the contract
    # TODO: we need bondTotalDuration in bond repayment in blocks
    draft_response = save_draft(user_id, request)
    draft_id = draft_response["draft_id"]


    # Retrieve the draft bond data
    draft_data = projects.get(draft_id)
    if not draft_data or draft_data["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Draft not found or user mismatch")

    # Create data classes from the draft data
    project_info = convert_project_info_to_struct(draft_data["project_info"])
    bond_details = convert_bond_details_to_struct(draft_data["bond_details"])
    auction_schedule = convert_auction_schedule_to_struct(draft_data["auction_schedule"])
    bond_repayment = convert_bond_repayment_to_struct(draft_data["bond_repayment"])
    
    # Create the bond contract using the data classes
    bond_id, bond_contract = launcher_contract.create_bond_contract(
        user_id, project_info, bond_details, auction_schedule, bond_repayment
    )
    tokens[bond_contract.contract_address] = bond_contract
    # TODO: delete the draft

    return {"bond_id": bond_id}

def calculate_apr(future_value, present_value, days=0, months=0, years=0):
    if present_value == 0:
        return 0
    # Convert the total duration into years
    duration_years = years + (months / 12) + (days / 365)
    simple_interest = (future_value - present_value) / present_value
    # Calculate APR
    apr = (simple_interest/duration_years) * 100
    return apr

def calculate_apr_from_bond(bond: BondContract):
    status = bond.get_bond_status()
    apr = convert_to_float(bond.bond_details.interestRate)
    if status == BondState.AUCTION_LIVE:
        auction_price = bond.get_current_auction_price()
        token_price = bond.bond_details.tokenPrice
        days = bond.bond_repayment.bondDuration.days
        months = bond.bond_repayment.bondDuration.months
        years = bond.bond_repayment.bondDuration.years
        apr = calculate_apr(token_price, auction_price, days, months, years)
        apr += convert_to_float(bond.bond_details.interestRate)
    return apr


def convert_bond_status_to_string(status: BondState):
    if status == BondState.AUCTION_NOT_STARTED:
        return "Pre-Auction"
    if status == BondState.AUCTION_LIVE:
        return "Auction Live"
    if status == BondState.BOND_LIVE:
        return "Bond Live"
    if status == BondState.BOND_ENDED:
        return "Bond Ended"
    if status == BondState.BOND_CANCELLED:
        return "Bond Cancelled"
    if status == BondState.ACTIVITY_HALTED:
        return "Activity Halted"

def build_bond_preview_from_bond(bond: BondContract):

    status = bond.get_bond_status()
    apr = calculate_apr_from_bond(bond)
    if bond.auction_end_block != 0:
        bond_end_date = convert_blocks_to_date_str(bond.auction_end_block + bond.bond_repayment.bondTotalDurationBlocks)
    else:
        bond_end_date = ""
    
    return PublishedBondPreview(
        contract_address=bond.contract_address,
        name=bond.bond_details.title,
        status=convert_bond_status_to_string(status),
        apr=apr,
        interest=convert_to_float(bond.bond_details.interestRate),
        duration=BondDuration(**asdict(bond.bond_repayment.bondDuration)),
        token_price=convert_to_float(bond.bond_details.tokenPrice),
        tokens=convert_to_float(bond.bond_details.tokens),
        total_amount=convert_to_float(bond.bond_details.totalAmount),
        total_supply=convert_to_float(bond.total_supply),
        auction_start_block=bond.auction_start_block,
        auction_start_date=convert_blocks_to_date_str(bond.auction_start_block),
        auction_end_block=bond.auction_end_block,
        auction_end_date=convert_blocks_to_date_str(bond.auction_end_block),
        next_snap_shot_block=bond.next_eligible_snapshot(),
        image_url=bond.project_info.imageUrl,
        late_penalty=convert_to_float(bond.bond_repayment.latePenalty),
        bond_end_date=bond_end_date
    )


def convert_project_info_to_struct(basemodel_instance):
    return ProjectInfoStruct(**basemodel_instance.dict())

def convert_bond_details_to_struct(basemodel_instance):
    bond_details_dict = basemodel_instance.dict()
    bond_details_dict['tokenPrice'] = convert_to_integer(bond_details_dict['tokenPrice'])
    bond_details_dict['interestRate'] = convert_to_integer(bond_details_dict['interestRate'])
    bond_details_dict['totalAmount'] = convert_to_integer(bond_details_dict['totalAmount'])
    bond_details_dict['tokens'] = convert_to_integer(bond_details_dict['tokens'])
    return BondDetailsStruct(**bond_details_dict)

def convert_auction_schedule_to_struct(basemodel_instance):
    auction_schedule_dict = basemodel_instance.dict()
    auction_schedule_dict['auctionDuration'] = AuctionDurationStruct(**auction_schedule_dict['auctionDuration'])
    auction_schedule_dict['adjustmentDetails'] = AdjustmentDetailsStruct(
        **{k: (convert_to_integer(v) if k in ['amount', 'rate'] else v) for k, v in auction_schedule_dict['adjustmentDetails'].items()}
    )
    auction_schedule_dict['minPrice'] = convert_to_integer(auction_schedule_dict['minPrice'])
    del auction_schedule_dict['startDate']
    # auction_schedule_dict['startBlock'] = convert_to_blocks(auction_schedule_dict['startDate'])
    return AuctionScheduleStruct(**auction_schedule_dict)

def convert_bond_repayment_to_struct(basemodel_instance):
    bond_repayment_dict = basemodel_instance.dict()
    bond_repayment_dict['bondDuration'] = BondDurationStruct(**bond_repayment_dict['bondDuration'])
    bond_repayment_dict['fixedPaymentInterval'] = FixedPaymentIntervalStruct(**bond_repayment_dict['fixedPaymentInterval'])
    bond_repayment_dict['customRepaymentSchedule'] = [
        CustomRepaymentIntervalStruct(
            **{k: (convert_to_integer(v) if k in ['principalPercent', 'interestPercent'] else v) for k, v in item.items()}
        ) for item in bond_repayment_dict['customRepaymentSchedule']
    ]
    bond_repayment_dict['latePenalty'] = convert_to_integer(bond_repayment_dict['latePenalty'])
    bond_repayment_dict['bondTotalDurationBlocks'] = convert_date_to_blocks(**asdict(bond_repayment_dict['bondDuration']))
    return BondRepaymentStruct(**bond_repayment_dict)


def convert_project_dataclass_to_info_basemodel(dataclass_instance):
    return ProjectInfo(**asdict(dataclass_instance))

def convert_bond_dataclass_to_details_basemodel(dataclass_instance):
    bond_details_dict = asdict(dataclass_instance)
    bond_details_dict['tokenPrice'] = convert_to_float(bond_details_dict['tokenPrice'])
    bond_details_dict['interestRate'] = convert_to_float(bond_details_dict['interestRate'])
    bond_details_dict['totalAmount'] = convert_to_float(bond_details_dict['totalAmount'])
    bond_details_dict['tokens'] = convert_to_float(bond_details_dict['tokens'])
    return BondDetails(**bond_details_dict)

def convert_auction_dataclass_to_schedule_basemodel(dataclass_instance):
    auction_schedule_dict = asdict(dataclass_instance)
    auction_schedule_dict['auctionDuration'] = AuctionDuration(**auction_schedule_dict['auctionDuration'])
    auction_schedule_dict['adjustmentDetails'] = AdjustmentDetails(
        **{k: (convert_to_float(v) if k in ['amount', 'rate'] else v) for k, v in auction_schedule_dict['adjustmentDetails'].items()}
    )
    
    auction_schedule_dict['startDate'] = convert_blocks_to_date_str(auction_schedule_dict['startBlock'])

    auction_schedule_dict['minPrice'] = convert_to_float(auction_schedule_dict['minPrice'])
    
    auction_schedule = AuctionSchedule(**auction_schedule_dict)
    return auction_schedule

def convert_bond_dataclass_to_repayment_basemodel(dataclass_instance):
    bond_repayment_dict = asdict(dataclass_instance)
    bond_repayment_dict['bondDuration'] = BondDuration(**bond_repayment_dict['bondDuration'])
    bond_repayment_dict['fixedPaymentInterval'] = FixedPaymentInterval(**bond_repayment_dict['fixedPaymentInterval'])
    bond_repayment_dict['customRepaymentSchedule'] = [
        CustomRepaymentInterval(
            **{k: (convert_to_float(v) if k in ['principalPercent', 'interestPercent'] else v) for k, v in item.items()}
        ) for item in bond_repayment_dict['customRepaymentSchedule']
    ]
    bond_repayment_dict['latePenalty'] = convert_to_float(bond_repayment_dict['latePenalty'])
    return BondRepayment(**bond_repayment_dict)


def build_bond_details_from_bond(bond: BondContract, user_id: str):
    amount_entitled_to = bond.get_amount_user_entitled_to(user_id)
    amount_user_entitled_to_and_claimable = []
    for elem in amount_entitled_to:
        record = AmountUserEntitledToRecord(
            snapshot_block=elem[0],
            principal_owed=convert_to_float(elem[1]),
            penalty_owed=convert_to_float(elem[2]),
            amount_already_claimed=convert_to_float(elem[3]),
            available_to_claim=convert_to_float(elem[4])
        )
        amount_user_entitled_to_and_claimable.append(record)

    total_issuer_owes = bond.get_total_owed()
    total_issuer_owes_break_down = []
    for elem in bond.get_total_owed_breakdown():
        record = IssuerOwedBreakdown(
            snapshot_block=elem[0],
            principal_owed=convert_to_float(elem[1]),
            penalty_owed=convert_to_float(elem[2])
        )
        total_issuer_owes_break_down.append(record)

    if bond.bond_end_block != 0:
        bond_end_date = convert_blocks_to_date_str(bond.bond_end_block)
    elif bond.auction_end_block != 0:
        bond_end_date = convert_blocks_to_date_str(bond.auction_end_block + bond.bond_repayment.bondTotalDurationBlocks)
    else:
        bond_end_date = ""

    return PublishedBondDetails(
        contract_address=bond.contract_address,
        issuer=bond.issuer,
        total_supply=convert_to_float(bond.total_supply),
        project_info=convert_project_dataclass_to_info_basemodel(bond.project_info),
        bond_details=convert_bond_dataclass_to_details_basemodel(bond.bond_details),
        auction_schedule=convert_auction_dataclass_to_schedule_basemodel(bond.auction_schedule),
        bond_repayment=convert_bond_dataclass_to_repayment_basemodel(bond.bond_repayment),
        activity_halted=bond.activity_halted,
        withdrawn_funds = bond.withdrawn_funds,
        funds_from_bond_purchase=convert_to_float(bond.funds_from_bond_purchase),
        auction_start_block=bond.auction_start_block,
        auction_end_block=bond.auction_end_block,
        bond_end_block=bond.bond_end_block,
        bond_manually_cancelled=bond.bond_manually_cancelled,
        bond_end_date=bond_end_date,
        auction_end_date=convert_blocks_to_date_str(bond.auction_end_block),
        overpayment=convert_to_float(bond.overpayment),
        current_block = get_current_block(),
        next_eligible_snapshot=bond.next_eligible_snapshot(),
        total_issuers_owes=TotalIssuerOwes(
            total_principal_owed=convert_to_float(total_issuer_owes[0]), 
            total_penalty_owed=convert_to_float(total_issuer_owes[1])
        ),
        total_issuer_owes_break_down=total_issuer_owes_break_down,
        amount_user_entitled_to_and_claimable=amount_user_entitled_to_and_claimable,
        remaining_tokens=convert_to_float(bond.get_remaining_tokens()),
        bond_sold_out=bond.is_bond_sold_out(),
        bond_status=convert_bond_status_to_string(bond.get_bond_status()),
        current_auction_price=convert_to_float(bond.get_current_auction_price()),
        apr=calculate_apr_from_bond(bond)
    )


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
    draft_request.bond_details.paymentTokenAddress = frax_token_id
    draft_id = draft_request.draft_id
    if draft_id and draft_id in projects and projects[draft_id]["user_id"] == user_id:
        # Update existing draft
        projects[draft_id].update({
            "project_info": draft_request.project_info,
            "bond_details": draft_request.bond_details,
            "auction_schedule": draft_request.auction_schedule,
            "bond_repayment": draft_request.bond_repayment,
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
            "bond_repayment": draft_request.bond_repayment,
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

@app.post("/purchase_bond/{bond_id}")
def purchase_bond(bond_id: str, request: PurchaseRequest):
    user_id = request.user_id
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")
    
    bond_contract:BondContract = launcher_contract.bonds[bond_id]
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
def withdraw_funds(user_id: str, bond_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract: BondContract = launcher_contract.bonds[bond_id]
    if bond_contract.issuer != user_id:
        raise HTTPException(status_code=403, detail="Only the issuer can withdraw funds")

    if not bond_contract.withdraw_funds(user_id):
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

    bond_contract:BondContract = launcher_contract.bonds[bond_id]
    try:
        total_claimed = bond_contract.claim_payments(holder)
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

@app.get("/bonds", response_model=List[PublishedBondPreview])
def get_bonds():
    if len(launcher_contract.bonds) == 0:
        return list()
    bond_previews = [build_bond_preview_from_bond(bond) for bond in launcher_contract.bonds.values()]
    return list(bond_previews)

@app.get("/bond/{bond_id}", response_model=PublishedBondDetails)
def get_bond_details(bond_id: str, user_id:str):
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")

    bond_contract = launcher_contract.bonds[bond_id]
    return build_bond_details_from_bond(bond_contract, user_id)

@app.get("/current_block", response_model=Dict)
def get_current_block_info():
    current_block = get_current_block()
    return {"current_block": current_block}

@app.post("/advance_current_block_by_blocks", response_model=Dict)
def advance_current_block_by_blocks(amount_to_advance: int):
    blockchain_state.advance_blocks(amount_to_advance)
    return {"new_current_block": blockchain_state.get_current_block()}

@app.post("/advance_current_block_by_days", response_model=Dict)
def advance_current_block_by_days(update_blocks_by_days_request: UpdateBlocksByDaysRequest):
    blocks_to_advance = convert_days_to_blocks(float(update_blocks_by_days_request.days_to_advance))
    blockchain_state.advance_blocks(blocks_to_advance)
    return {"new_current_block": blockchain_state.get_current_block()}

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
def create_snapshot(bond_id:str, request: CreateSnapshotRequest):
    user_id = request.user_id
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract: BondContract = launcher_contract.bonds[bond_id]
    try:
        snapshot_block = bond_contract.create_snapshot(user_id)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Snapshot created successfully", "snapshot_block": snapshot_block}

@app.post("/bonds/{bond_id}/deposit_payment")
def deposit_payment(bond_id: str, amount: float):
    if bond_id not in launcher_contract.bonds:
        raise HTTPException(status_code=404, detail="Bond not found")

    bond_contract: BondContract = launcher_contract.bonds[bond_id]
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

@app.get("/balance/{tokenAddress}")
def get_balance(tokenAddress: str, user_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if tokenAddress not in tokens:
        raise HTTPException(status_code=404, detail="Token not found")
    token_contract = tokens[tokenAddress]
    balance = convert_to_float(token_contract.balanceOf(user_id))
    return {"balance": balance}

@app.post("/approve/{tokenAddress}")
def approve(tokenAddress: str, request: ApproveRequest):
    if request.user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if tokenAddress not in tokens:
        raise HTTPException(status_code=404, detail="Token not found")
    user_id = request.user_id
    spender = request.spender
    amount = request.amount
    token_contract: ERC20Token = tokens[tokenAddress]
    token_contract.approve(user_id, spender, convert_to_integer(amount))
    return {"message": "Approval set successfully"}

@app.get("/approved/{tokenAddress}")
def get_approved(tokenAddress: str, user_id: str, spender_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if tokenAddress not in tokens:
        raise HTTPException(status_code=404, detail="Token not found")
    token_contract = tokens[tokenAddress]
    approved = token_contract.allowanceOf(user_id, spender_id)
    return {"approvedAmount": convert_to_float(approved)}

@app.post("/swap", response_model=SwapResponse)
def swap(request: SwapRequest):
    if request.user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if request.sendTokenAddress not in tokens:
        raise HTTPException(status_code=404, detail="Token not found")
    if request.receiveTokenAddress not in tokens:
        raise HTTPException(status_code=404, detail="Token not found")

    sendTokenAddress = request.sendTokenAddress
    receiveTokenAddress = request.receiveTokenAddress
    sendAmount = request.sendAmount
    receiveAmount = request.receiveAmount
    user_id = request.user_id

    sendTokenContract = tokens[sendTokenAddress]
    receiveTokenContract = tokens[receiveTokenAddress]

    

# Run the server using the command: uvicorn main:app --reload
