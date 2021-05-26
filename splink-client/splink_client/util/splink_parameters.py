"""
    sPLINK-specific server and client parameters

    Copyright 2021 Reza NasiriGerdeh and Reihaneh TorkzadehMahani. All Rights Reserved.

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


class SplinkProjectParameter:
    COVARIATES = "covariates"
    CHUNK_SIZE = "chunk_size"
    MAX_ITERATIONS = "max_iterations"


class SplinkLocalParameter:
    SAMPLE_COUNT = 'sample_count'
    SNP_ID = 'local_snp_id'
    ALLELE_NAME = 'allele_name'
    NON_MISSING_SAMPLE_COUNT = 'non_missing_sample_count'
    ALLELE_COUNT = 'allele_count'
    CONTINGENCY_TABLE = 'contingency_table'
    XT_X_MATRIX = 'xt_x_matrix'
    XT_Y_VECTOR = 'xt_y_vector'
    SSE = 'sse'
    GRADIENT = "gradient"
    HESSIAN = "hessian"
    LOG_LIKELIHOOD = "log_likelihood"


class SplinkGlobalParameter:
    SNP_ID = 'global_snp_id'
    CURRENT_CHUNK = "current_chunk"
    TOTAL_CHUNKS = "total_chunks"
    CHUNK_START_INDEX = "chunk_start_index"
    CHUNK_END_INDEX = "chunk_end_index"
    SNP_INDEX = "snp_index"
    MINOR_ALLELE_NAME = "minor_allele_name"
    MAJOR_ALLELE_NAME = "major_allele_name"
    BETA = "beta"
    CURRENT_BETA_ITERATION = "current_beta_iteration"
