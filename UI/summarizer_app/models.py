from django.db import models

# Create your models here.
class Summary(models.Model):
    pmc_id = models.CharField(max_length=12, primary_key=True)
    basic_summary = models.TextField()
    college_summary= models.TextField()
    professional_summary = models.TextField()
