from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
import json
import numpy as np
import pandas as pd

from keras.models import load_model
from rest_framework import viewsets

from .serializers import SampleSerializer
from .models import Sample


class SampleViewSet(viewsets.ModelViewSet):
    queryset = Sample.objects.all()
    serializer_class = SampleSerializer

@api_view(["POST"])
def approvereject(request):
    try:
        model = load_model('model.h5')
        mydata=request.data
        unit=np.array(list(mydata.values()))
        X=unit.reshape(1,3)
        pred = np.sum(model.predict(X))

        newdf=pd.DataFrame(pred, columns=['Status'])
        return JsonResponse('Predict is {}'.format(newdf), safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)