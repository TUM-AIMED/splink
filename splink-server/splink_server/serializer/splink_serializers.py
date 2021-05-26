"""
    sPLINK project serializer to serialize project specific fields

    Copyright 2021 Reihaneh TorkzadehMahani and Reza NasiriGerdeh. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from rest_framework import serializers
from hyfed_server.serializer.hyfed_serializers import HyFedProjectSerializer


class SplinkProjectSerializer(HyFedProjectSerializer):
    """ Serializes the sPLINK project model to serve a WebApp/client request """

    chunk_size = serializers.SerializerMethodField()
    covariates = serializers.SerializerMethodField()
    max_iterations = serializers.SerializerMethodField()

    def get_chunk_size(self, instance):
        return instance.chunk_size

    def get_covariates(self, instance):
        return instance.covariates

    def get_max_iterations(self, instance):
        return instance.max_iterations

    class Meta(HyFedProjectSerializer.Meta):
        fields = HyFedProjectSerializer.Meta.fields + ('chunk_size', 'covariates', 'max_iterations',)
