from django.db import models

# Create your models here.
class Sample(models.Model):
    X1 = models.FloatField(default=0.0)
    X2 = models.FloatField(default=0.0)
    X3 = models.FloatField(default=0.0)