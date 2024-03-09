from django.db import models

class TemperaturePrediction(models.Model):
    city = models.CharField(max_length=255)
    year = models.IntegerField()
    month = models.IntegerField()
    max_temperature = models.FloatField()
    min_temperature = models.FloatField()
    city_prediction = models.CharField(max_length=255)

    def __str__(self):
        return f'{self.city} - {self.year}/{self.month}'