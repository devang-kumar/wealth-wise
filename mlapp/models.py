from django.db import models
from django.contrib.auth.models import User

class FinancialPrediction(models.Model):
    QUALIFICATION_CHOICES = [
        ('Graduate', 'Graduate'),
        ('Post Graduate', 'Post Graduate'),
        ('Professional', 'Professional'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    monthly_income = models.DecimalField(max_digits=12, decimal_places=2)
    family_members = models.IntegerField()
    emi_rent = models.DecimalField(max_digits=12, decimal_places=2)
    annual_income = models.DecimalField(max_digits=12, decimal_places=2)
    qualification = models.CharField(max_length=20, choices=QUALIFICATION_CHOICES)
    earning_members = models.IntegerField()
    predicted_spending = models.DecimalField(max_digits=12, decimal_places=2)
    predicted_savings = models.DecimalField(max_digits=12, decimal_places=2)
    savings_percentage = models.DecimalField(max_digits=5, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username}'s prediction on {self.created_at}"