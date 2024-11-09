from django.db import models
import os

# Create your models here.
class dataset(models.Model):
    data = models.FileField(upload_to='diabetes.csv')

    def filename(self):
        return os.path.basename(self.data.name)