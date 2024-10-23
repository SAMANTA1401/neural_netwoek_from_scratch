#Before SOLID (Violations):
class PaymentProcessor:
    def process_payment(self, amount, payment_method):
        if payment_method == "credit_card":
            # Credit card processing logic
            print("Processing credit card payment")
        elif payment_method == "paypal":
            # PayPal processing logic
            print("Processing PayPal payment")
        else:
            raise ValueError("Unsupported payment method")

payment_processor = PaymentProcessor()
payment_processor.process_payment(100, "credit_card")



# After SOLID (Compliance):
from abc import ABC, abstractmethod

# SRP: Separate classes for each payment method
class PaymentMethod(ABC):
    @abstractmethod
    def process_payment(self, amount):
        pass

class CreditCardPaymentMethod(PaymentMethod):
    def process_payment(self, amount):
        print("Processing credit card payment")

class PayPalPaymentMethod(PaymentMethod):
    def process_payment(self, amount):
        print("Processing PayPal payment")

# OCP: Open for extension, closed for modification
class PaymentProcessor:
    def __init__(self, payment_method):
        self.payment_method = payment_method

    def process_payment(self, amount):
        self.payment_method.process_payment(amount)

# LSP: Substitution principle
class PaymentProcessorFactory:
    def create_payment_processor(self, payment_method_type):
        if payment_method_type == "credit_card":
            return PaymentProcessor(CreditCardPaymentMethod())
        elif payment_method_type == "paypal":
            return PaymentProcessor(PayPalPaymentMethod())
        else:
            raise ValueError("Unsupported payment method")

# ISP: Client (PaymentProcessor) depends on interface (PaymentMethod)
# DIP: High-level module (PaymentProcessor) depends on abstraction (PaymentMethod)
payment_processor_factory = PaymentProcessorFactory()
payment_processor = payment_processor_factory.create_payment_processor("credit_card")
payment_processor.process_payment(100)