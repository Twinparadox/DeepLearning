from rest_framework import serializers

from .models import Sample

class SampleSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Sample
        fields = ('X1', 'X2', 'X3')