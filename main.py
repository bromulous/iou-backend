from fastapi.exceptions import RequestValidationError
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import datetime
import secrets
from dataclasses import dataclass
from enum import Enum

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

class BondState(Enum):
    AUCTION_NOT_STARTED = 0
    AUCTION_LIVE = 1
    BOND_LIVE = 2
    ACTIVITY_HALTED = 3
    BOND_CANCELLED = 4
    BOND_ENDED = 5

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

    def create_bond_contract(self, issuer, project_info, bond_details, auction_schedule, bond_repayment):
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

# Classes for ERC20Token and Bond operations
class ERC20Token:
    def __init__(self, name, symbol, total_supply):
        self.contract_address = generate_ethereum_address()
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
    def __init__(self, issuer, project_info, bond_details, auction_schedule, bond_repayment):
        super().__init__(bond_details.title, bond_details.tokenSymbol, 0)
        self.issuer = issuer
        self.project_info = project_info
        self.bond_details = bond_details
        self.auction_schedule = auction_schedule
        self.bond_repayment = bond_repayment
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
            start_block = self.snapshots[-1]
        end_snapshot = self.auction_end_block + self.bond_repayment.bondTotalDurationBlocks
        snapshot_block = min(snapshot_block, end_snapshot)
        if start_block >= snapshot_block:
            return 0
        total_amount = self.total_supply * self.bond_details.token_price
        if self.bond_repayment.paymentSchedule == "fixed":
            interest_due = self._get_apr_amount_for_block_window(
                total_amount=total_amount,
                interest_rate=self.bond_details.interest_rate,
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
        if curr_block <= next_snapshot_block:
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
    
    def _add_payment(self, payment):
        self.payments.append(payment)
        self.payments_index[payment.snapshot_block] = len(self.payments) - 1
    
    def _get_payment_for_block(self, snapshot_block):
        if snapshot_block not in self.payments_index:
            raise RuntimeError("Payment not found.")
        return self.payments[self.payments_index[snapshot_block]]
    
    def _get_apr_per_block(self, total_amount, interest_rate):
        apr_amount = (interest_rate * total_amount) // 100
        return apr_amount // (5760 * 365)
    
    def _get_apr_amount_for_block_window(self, total_amount, interest_rate, start_block, end_block):
        apr_per_block = self._get_apr_per_block(total_amount, interest_rate)
        if start_block >= end_block:
            raise ValueError("Start block must be less than end block.")
        elapsed_blocks = end_block - start_block
        return elapsed_blocks * apr_per_block
    
    def get_amount_owed_for_snapshot(self, snapshot_block):
        if snapshot_block not in self.snapshot_balances:
            return 0
        curr_block = get_current_block()

        payment_info = self._get_payment_for_block(snapshot_block)
        amount_due = payment_info.total_amount - payment_info.total_allocated
        if amount_due <= 0:
            return 0
        payment_due_block = snapshot_block + 5760 * 7 
        penalty_amount_remaining = 0
        if curr_block > payment_due_block and self.bond_details.latePenalty > 0:
            # The issuer has 7 days to allocate funds to the snapshot
            # If they don't allocate funds within this time, the penalty is applied
            penalty_amount = self._get_apr_amount_for_block_window(
                total_amount= amount_due,
                interest_rate=self.bond_details.latePenalty,
                start_block=payment_due_block,
                end_block=curr_block
            )
            penalty_amount_remaining = penalty_amount - payment_info.penalty_paid
        return (amount_due, penalty_amount_remaining)
    
    def get_total_owed(self):
        total_principal_owed = 0
        total_penalty_owed = 0
        for i in range(self.last_fully_paid_index + 1, len(self.payments)):
            payment = self.payments[i]
            principal, penalty = self.get_amount_owed_for_snapshot(payment["snapshot_block"])
            total_principal_owed += principal
            total_penalty_owed += penalty
        return (total_principal_owed, total_penalty_owed)
    
    def get_total_owed_breakdown(self):
        total_owed = []
        for i in range(self.last_fully_paid_index + 1, len(self.payments)):
            payment = self.payments[i]
            principal, penalty = self.get_amount_owed_for_snapshot(payment["snapshot_block"])
            total_owed += [[payment["snapshot_block"], principal, penalty]]
        return total_owed
    
    @reentrancy_protection
    def deposit_payment(self, amount):
        status = self.get_bond_status()
        if status != BondState.BOND_LIVE:
            raise RuntimeError("Bond is not live.")
        
        # Access the token instance using the token address
        payment_token_instance = tokens[self.bond_details.paymentTokenAddress]
        
        # Ensure the issuer has approved enough tokens for the contract to transfer
        if payment_token_instance.allowance(self.issuer, self.bond_id) < amount:
            raise RuntimeError("Insufficient token approval.")
        
        # Transfer the tokens from issuer to contract
        # This might create an over payment situation
        # The overpayment will be allocated to the next snapshot
        # If the issuer realizes they have overpaid they can withdraw unallocated funds
        if not payment_token_instance.transferFrom(self.issuer, self.bond_id, amount):
            raise RuntimeError("Transfer failed.")
        
        self._allocate_payments(amount)
        
        
    
    def _allocate_payments(self, amount):
        amount += self.overpayment
        
        repaid = 0
        
        for i in range(self.last_fully_paid_index + 1, len(self.payments)):
            payment = self.payments[i]
            principal_due, penalty_due = self.get_amount_owed_for_snapshot(payment["snapshot_block"])
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
        if self.total_repaid >= self.total_supply * self.bond_details.tokenPrice:
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
            return 0
        
        payment_info = self._get_payment_for_block(snapshot_block)
        principal_percentage_owed = (payment_info.total_allocated * balance_at_snapshot) // self.total_supply
        penalty_percentage_owed = (payment_info.penalty_paid * balance_at_snapshot) // self.total_supply
        amount_already_claimed = payment_info.balances.get(holder, 0)
        available_to_claim = principal_percentage_owed + penalty_percentage_owed - amount_already_claimed

        return [principal_percentage_owed, penalty_percentage_owed, amount_already_claimed, available_to_claim]
    
    def get_amount_user_entitled_to(self, holder):
        amounts = []
        for i in range(len(self.payments)):
            payment = self.payments[i]
            snapshot_block = payment["snapshot_block"]
            amounts += [[snapshot_block] + self.get_amount_user_entitled_to_for_snapshot(holder, snapshot_block)]

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
            balance_at_snapshot = self.balance_of_at(holder, payment["snapshot_block"])
            if balance_at_snapshot == 0:
                continue
            principal_percentage_owed = (payment.total_allocated * balance_at_snapshot) // self.total_supply
            penalty_percentage_owed = (payment.penalty_paid * balance_at_snapshot) // self.total_supply
            amount_already_claimed = payment.balances.get(holder, 0)
            available_to_claim = principal_percentage_owed + penalty_percentage_owed - amount_already_claimed
            if available_to_claim == 0:
                continue
            payment.balances[holder] += available_to_claim
            payment.total_claimed += available_to_claim
            total_claimed += available_to_claim
        # Transfer token to holder
        payment_token_instance = tokens[self.bond_details.paymentTokenAddress]
        if not payment_token_instance.transfer(self.bond_id, holder, total_claimed):
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
            and self.auction_end_block != 0 and curr_block <= self.auction_end_block
    
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

    
    def get_bond_status(self):
        # states:
        # 0 - auction not started
        # 1 - auction started
        # 2 - auction ended/bond is live
        # 3 - activity halted
        # 4 - bond cancelled
        current_block = get_current_block()
        passed_auction_start_block = False
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
        if self._is_bond_cancelled():
            return BondState.BOND_CANCELLED
        

    @reentrancy_protection
    def purchase_bond(self, buyer, payment_token_amount, bond_token_amount):
        auction_price = self.get_current_auction_price()
        status = self.get_bond_status()
        if status != BondState.AUCTION_LIVE:
            raise RuntimeError("Auction is not live.")

        expected_bond_token_amount = payment_token_amount // auction_price
        if bond_token_amount < expected_bond_token_amount:
            raise ValueError("The number of bond tokens does not match the payment amount submitted at the current price.")
        
        bond_tokens_to_transfer = min(bond_token_amount, expected_bond_token_amount)
        
        if not self.bond_details.infiniteTokens and self.bond_details.tokens - self.total_supply < bond_tokens_to_transfer:
            raise RuntimeError("Insufficient bond inventory.")

        total_cost = bond_tokens_to_transfer * auction_price
        
        payment_token = tokens[self.bond_details.paymentTokenAddress]
        if payment_token.allowances.get(buyer, {}).get(self.bond_id, 0) < total_cost:
            raise RuntimeError("Insufficient token approval.")
        
        if not payment_token.transferFrom(buyer, self.bond_id, total_cost):
            raise RuntimeError("Transfer failed.")
        
        self.total_supply += bond_tokens_to_transfer
        self.balances[buyer] += bond_tokens_to_transfer
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
        funds_token = tokens[self.bond_details.paymentTokenAddress]
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
                price_interval_delta = price * self.auction_schedule.AdjustmentDetailsStruct.rate
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

class PaymentScheduleItem(BaseModel):
    days: int
    months: int
    years: int
    amount: float

class Bond(BaseModel):
    bond_id: str

class IssueBondRequest(BaseModel):
    user_id: str
    draft_id: str


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
        if self.startDate != "":
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

class BondDetails(BaseModel):
    title: str
    totalAmount: int
    infiniteTokens: bool
    tokens: int
    tokenPrice: float
    tokenSymbol: str
    interestRate: float
    requiresFullSale: bool
    latePenalty: float
    earlyRepayment: bool
    collateral: bool
    paymentTokenAddress: Optional[str] = frax_token_id

class BondRepayment(BaseModel):
    bondDuration: BondDuration
    repaymentType: str
    paymentSchedule: str
    fixedPaymentInterval: FixedPaymentInterval
    customRepaymentSchedule: List[CustomRepaymentInterval]

class SaveDraftRequest(BaseModel):
    draft_id: Optional[str] = None
    project_info: ProjectInfo
    bond_details: BondDetails
    auction_schedule: AuctionSchedule
    bond_repayment: BondRepayment

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
    latePenalty: int
    earlyRepayment: bool
    collateral: bool
    paymentTokenAddress: str

@dataclass
class BondRepaymentStruct:
    bondDuration: BondDurationStruct
    bondTotalDurationBlocks: int
    repaymentType: str
    paymentSchedule: str
    fixedPaymentInterval: FixedPaymentIntervalStruct
    customRepaymentSchedule: List[CustomRepaymentIntervalStruct]

@dataclass
class SaveDraftRequestStruct:
    project_info: ProjectInfoStruct
    bond_details: BondDetailsStruct
    auction_schedule: AuctionScheduleStruct
    bond_repayment: BondRepaymentStruct
    draft_id: Optional[str] = None

# Helper Conversion Functions
def convert_to_integer(amount: float, multiplier: int = 10000) -> int:
    return int(amount * multiplier)

def convert_to_blocks(date: datetime) -> int:
    # Assume each day is 5760 blocks (15 seconds per block)
    return (date.date() - datetime.date(1970, 1, 1)).days * 5760

def convert_date_to_blocks(hours=0, days = 0, months = 0, years = 0):
    # Assume each day is 5760 blocks (15 seconds per block)
    return (years * 365 * 5760) + (months * 30 * 5760) + (days * 5760) + (hours * 240)

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
    # user_bonds = [launcher_contract.bonds[bond_id] for bond_id in launcher_contract.issuer_bonds.get(user_id, [])]
    # user["bonds_created"] = user_bonds
    user["bonds_created"] = [{'bond_id': id} for id in launcher_contract.issuer_bonds.get(user_id, [])]
    
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
def issue_bond(user_id: str, request: IssueBondRequest):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if launcher_contract.bond_issuance_stopped:
        raise HTTPException(status_code=403, detail="Bond issuance has been halted")
    
    # TODO: auction startDate needs to be converted to block number and changed to startBlock
    # TODO: Currently auction only ends based on whether it is sold out or manually ended
    # TODO: need an auction end data and convert to block number
    # TODO: timezone is not used in the contract
    # TODO: we need bondTotalDuration in bond repayment in blocks

    # Retrieve the draft bond data
    draft_data = projects.get(request.draft_id)
    if not draft_data or draft_data["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Draft not found or user mismatch")

    # Create data classes from the draft data
    project_info = ProjectInfoStruct(**draft_data["project_info"].dict())
    bond_details = BondDetailsStruct(**draft_data["bond_details"].dict())
    auction_schedule = AuctionScheduleStruct(
        auctionDuration=AuctionDurationStruct(**draft_data["auction_schedule"].dict()["auctionDuration"]),
        startBlock=draft_data["auction_schedule"].startBlock,
        endBlock= draft_data["auction_schedule"].endBlock,
        **{k: v for k, v in draft_data["auction_schedule"].dict().items() if k not in ("auctionDuration", "startDate")}
    )
    bond_repayment = BondRepaymentStruct(
        bondDuration=BondDurationStruct(**draft_data["bond_repayment"].dict()["bondDuration"]),
        bondTotalDurationBlocks=convert_date_to_blocks(**draft_data["bond_repayment"].dict()["bondDuration"]),
        fixedPaymentInterval=FixedPaymentIntervalStruct(**draft_data["bond_repayment"].dict()["fixedPaymentInterval"]),
        customRepaymentSchedule=[
            CustomRepaymentIntervalStruct(**item) for item in draft_data["bond_repayment"].dict()["customRepaymentSchedule"]
        ],
        **{k: v for k, v in draft_data["bond_repayment"].dict().items() if k not in ["bondDuration", "fixedPaymentInterval", "customRepaymentSchedule"]}
    )

    # Create the bond contract using the data classes
    bond_id, bond_contract = launcher_contract.create_bond_contract(
        user_id, project_info, bond_details, auction_schedule, bond_repayment
    )
    # TODO: delete the draft

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
