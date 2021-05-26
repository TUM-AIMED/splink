"""
    Client-side sPLINK project to compute local parameters

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

from hyfed_client.project.hyfed_client_project import HyFedClientProject
from hyfed_client.util.hyfed_steps import HyFedProjectStep
from hyfed_client.util.operation import ClientOperation

from splink_client.util.splink_steps import SplinkProjectStep
from splink_client.util.splink_algorithms import SplinkAlgorithm
from splink_client.util.splink_parameters import SplinkGlobalParameter, SplinkLocalParameter, SplinkProjectParameter
from splink_client.util.gwas_dataset import GwasDataset, PhenotypeValue, SnpValue, MissingValue

import numpy as np
import threading
import multiprocessing


class SplinkClientProject(HyFedClientProject):
    """
        A class that provides computation functions to compute local parameters for chi-square, linear/logistic regression
    """

    def __init__(self, username, token, project_id, server_url, compensator_url,
                 algorithm, name, description, coordinator, result_dir, log_dir,
                 dataset_file_path, phenotype_file_path, covariate_file_path,  # Splink specific arguments
                 phenotype_name, covariate_names, chunk_size, max_iterations, cpu_cores):  # Splink specific arguments

        super().__init__(username=username, token=token, project_id=project_id, server_url=server_url, compensator_url=compensator_url,
                         algorithm=algorithm, name=name, description=description, coordinator=coordinator,
                         result_dir=result_dir, log_dir=log_dir)

        # Splink project (hyper-)parameters
        self.chunk_size = chunk_size * 1000
        self.max_iterations = max_iterations

        # dataset (fam/bim/bed), phenotype, and covariate file paths
        self.dataset_file_path = dataset_file_path
        self.phenotype_file_path = phenotype_file_path
        self.phenotype_name = phenotype_name

        self.covariate_file_path = covariate_file_path
        self.covariate_names = tuple(
            [cov_name.strip() for cov_name in covariate_names.split(',')]) if covariate_names else ()

        self.cpu_cores = cpu_cores

        # ### attributes to compute local parameters; re-initialized in init_step function
        # fam file related attributes
        self.sex_values = np.array([])
        self.phenotype_values = np.array([])
        self.sample_count = 0

        # bim file related attributes
        self.snp_id_values = np.array([])  # chromosome_number, snp_name, and base_pair_position combined
        self.first_allele_names = dict()  # indexed by the SNP ID
        self.second_allele_names = dict()  # indexed by the SNP ID

        # bed file related attributes
        self.snp_values = dict()  # indexed by the SNP ID; in sample_count_step, it is converted to a list

        # covariate file related attribute
        self.covariate_values = dict()

        # attributes to speed-up the creation of the feature matrix
        self.non_missing_index_values = np.array([])
        self.covariate_matrix = np.array([])

        # chunk attributes;
        self.current_chunk = -1  # re-initialized in the non_missing_count_step function
        self.total_chunks = -1  # re-initialized in the non_missing_count_step function
        self.sub_chunk_start_indices = list()  # re-initialized in the set_sub_chunk_indices function
        self.sub_chunk_end_indices = list()  # re-initialized in the set_sub_chunk_indices function
        self.snp_indices = set()  # re-initialized in each step of the project
        self.current_chunk_size = 0  # it is always len(self.snp_indices)

        # ### The following dictionaries are indexed by the SNP index

        # attributes for non-missing-count step
        self.non_missing_sample_counts = dict()  # number of samples, where none of the covariate, sex, and SNP values is missing
        self.allele_counts = dict()  # number of first/second alleles of each SNP

        # contingency table for chi-square test
        self.contingency_tables = dict()

        # linear regression related dictionaries
        self.xt_x_matrix = dict()
        self.xt_y_vector = dict()
        self.sse_values = dict()  # sum square error values

        # logistic regression related functions
        self.current_beta_iteration = -1
        self.gradient_vectors = dict()
        self.hessian_matrices = dict()
        self.log_likelihood_values = dict()

    # ########## log functions
    def log_project_info(self):
        """ OVERRIDDEN: log sPLINK project general info """

        super().log_project_info()

        # sPLINK-specific parameters
        dataset_file = self.dataset_file_path.split("/")[-1][:-4]
        self.log(f"Dataset file: {dataset_file}", include_date=False)

        if self.phenotype_file_path:
            phenotype_file = self.phenotype_file_path.split("/")[-1]
            self.log(f'Phenotype file: {phenotype_file}', include_date=False)
        else:
            self.log(f'Phenotype file: -', include_date=False)

        if self.covariate_file_path:
            covariates_file = self.covariate_file_path.split("/")[-1]
            self.log(f'Covariates file: {covariates_file}', include_date=False)
            covariate_names = ','.join(self.covariate_names)
            self.log(f'Covariate names: {covariate_names}', include_date=False)
        else:
            self.log(f'Covariates file: -', include_date=False)

        self.log(f'CPU cores: {self.cpu_cores}', include_date=False)

        self.log("\n", include_date=False)

    # ########## Splink step functions
    def init_step(self):
        """ Initialize the GWAS dataset """

        try:
            # ##### open GWAS dataset
            gwas_dataset = GwasDataset(bed_file_path=self.dataset_file_path,
                                       phenotype_file_path=self.phenotype_file_path,
                                       covariate_file_path=self.covariate_file_path,
                                       phenotype_name=self.phenotype_name,
                                       covariate_names=self.covariate_names)

            self.log("Opening and pre-processing the GWAS dataset ...")
            gwas_dataset.open_and_preprocess()
            if gwas_dataset.is_operation_failed():
                self.log(gwas_dataset.get_error_message())
                self.set_operation_status_failed()
                return

            # phenotypes should be binary for logistic regression and chi-square
            if not gwas_dataset.is_phenotype_binary() and (self.algorithm == SplinkAlgorithm.LOGISTIC_REGRESSION or
                                                           self.algorithm == SplinkAlgorithm.CHI_SQUARE):
                self.log(f"Phenotype values must be binary for {self.algorithm} tests!")
                self.set_operation_status_failed()
                return

            # phenotype values should be quantitative for linear regression
            if gwas_dataset.is_phenotype_binary() and self.algorithm == SplinkAlgorithm.LINEAR_REGRESSION:
                self.log(f"Phenotype values must be quantitative for {self.algorithm} tests!")
                self.set_operation_status_failed()
                return

            # log general info about the gwas dataset
            self.log(gwas_dataset.get_dataset_info(), include_date=False)

            # #### initialize attributes required to compute local parameters

            # initialize fam file related attributes
            self.sex_values = gwas_dataset.get_sex_values()
            self.phenotype_values = gwas_dataset.get_phenotype_values()
            self.sample_count = gwas_dataset.get_sample_count()

            # initialize bim file related attributes
            self.snp_id_values = gwas_dataset.get_snp_id_values()
            self.first_allele_names = gwas_dataset.get_first_allele_names()
            self.second_allele_names = gwas_dataset.get_second_allele_names()

            # initialize bed file related attribute
            self.snp_values = gwas_dataset.get_snp_values()

            # initialize covariate file related attributes
            self.covariate_values = gwas_dataset.get_covariate_values()

            # initialize attributes to speed-up the creation of the feature matrix
            self.non_missing_index_values = gwas_dataset.get_non_missing_index_values()
            self.covariate_matrix = gwas_dataset.get_covariate_matrix()

        except Exception as io_exception:
            self.log(io_exception)
            self.set_operation_status_failed()

    def snp_id_step(self):
        """ share SNP IDs with the server """

        try:
            # share the SNP IDs whose minor allele frequency is non-zero with the server
            non_zero_snp_ids = np.array([snp_id for snp_id in self.snp_id_values if self.first_allele_names[snp_id] != '0'])
            self.local_parameters[SplinkLocalParameter.SNP_ID] = non_zero_snp_ids

        except Exception as snp_exception:
            self.log(snp_exception)
            self.set_operation_status_failed()

    def allele_name_step(self):
        """ Initialize SNP (ID) values and first/second allele names based on global (common) SNP IDs first,
            and then, share allele names with the server """

        try:
            # update snp_id_values, snp_values, first/second allele names
            # based on the global SNP IDs by excluding those that do not exist in the other clients
            # The above-mentioned attributed are converted to list

            # update SNP IDs
            self.snp_id_values = self.global_parameters[SplinkGlobalParameter.SNP_ID]
            self.log(f"{len(self.snp_id_values)} SNPs are common among all clients")

            # update SNP values
            snp_values = list()
            for snp_id in self.snp_id_values:
                snp_values.append(self.snp_values[snp_id])
            self.snp_values = snp_values

            # update first/second allele names and initialize allele names (shared with server)
            first_allele_names = list()
            second_allele_names = list()
            allele_names = [[], []]

            for snp_id in self.snp_id_values:
                first_allele_names.append(self.first_allele_names[snp_id])
                second_allele_names.append(self.second_allele_names[snp_id])

                # sort allele names to prevent revealing which allele is minor or major to the server
                if self.first_allele_names[snp_id] < self.second_allele_names[snp_id]:
                    allele_names[0].append(self.first_allele_names[snp_id])
                    allele_names[1].append(self.second_allele_names[snp_id])
                else:
                    allele_names[0].append(self.second_allele_names[snp_id])
                    allele_names[1].append(self.first_allele_names[snp_id])

            self.first_allele_names = first_allele_names
            self.second_allele_names = second_allele_names

            # share allele names with the server
            self.local_parameters[SplinkLocalParameter.ALLELE_NAME] = allele_names

        except Exception as allele_name_exception:
            self.log(allele_name_exception)
            self.set_operation_status_failed()

    def sample_count_step(self):
        """  share noisy local sample count with the server and noise with compensator """

        try:
            self.local_parameters[SplinkLocalParameter.SAMPLE_COUNT] = self.sample_count
            self.set_compensator_flag()
        except Exception as sample_count_exception:
            self.log(sample_count_exception)
            self.set_operation_status_failed()

    # ##### non_missing_count step related function
    def non_missing_count_step(self):
        """ init chunk attributes and compute local non-missing sample/allele count for the chunk """

        try:
            # init chunk attributes
            self.current_chunk = self.global_parameters[SplinkGlobalParameter.CURRENT_CHUNK]
            self.total_chunks = self.global_parameters[SplinkGlobalParameter.TOTAL_CHUNKS]
            self.snp_indices = self.global_parameters[SplinkGlobalParameter.SNP_INDEX]  # SNP indices in the current chunk
            self.current_chunk_size = len(self.snp_indices)
            chunk_start_index = self.global_parameters[SplinkGlobalParameter.CHUNK_START_INDEX]
            chunk_end_index = self.global_parameters[SplinkGlobalParameter.CHUNK_END_INDEX]
            self.set_sub_chunk_indices(chunk_start_index, chunk_end_index)

            # init count dictionaries
            self.non_missing_sample_counts = dict()
            self.allele_counts = dict()

            # queues
            queue_non_missing = multiprocessing.Queue()
            queue_allele_counts = multiprocessing.Queue()

            # threads to read from the queues
            thread_read_non_missing_queue = threading.Thread(target=self.read_queue_non_missing, args=(queue_non_missing,))
            thread_read_non_missing_queue.daemon = True
            thread_read_non_missing_queue.start()

            thread_read_allele_counts_queue = threading.Thread(target=self.read_queue_allele_counts, args=(queue_allele_counts,))
            thread_read_allele_counts_queue.daemon = True
            thread_read_allele_counts_queue.start()

            # start processes to compute the local non-missing sample counts as well as first/second allele counts for sub-chunks
            process_list = list()
            for start_index_sub_chunk, end_index_sub_chunk in zip(self.sub_chunk_start_indices, self.sub_chunk_end_indices):
                process = multiprocessing.Process(target=self.compute_non_missing_counts,
                                                  args=(start_index_sub_chunk, end_index_sub_chunk,
                                                        queue_non_missing, queue_allele_counts,))
                process_list.append(process)
                process.daemon = True
                process.start()

            # wait for read threads to be done
            thread_read_non_missing_queue.join()
            thread_read_allele_counts_queue.join()

            # close queues
            queue_non_missing.close()
            queue_allele_counts.close()

            # terminate the processes
            for proc in process_list:
                proc.terminate()

            # convert dictionaries to lists;
            # IMPORTANT: sorted(self.snp_indices) should always be used to ensure the order between list and set related SNP indices
            non_missing_sample_counts = list()
            allele_counts = list()
            for snp_index in sorted(self.snp_indices):
                non_missing_sample_counts.append(self.non_missing_sample_counts[snp_index])
                allele_counts.append(self.allele_counts[snp_index])

            # python list of scalars must be converted to a numpy array if compensator flag is set
            non_missing_sample_counts = np.array(non_missing_sample_counts)

            # share the noisy counts with the server and noise with compensator
            self.local_parameters[SplinkLocalParameter.NON_MISSING_SAMPLE_COUNT] = non_missing_sample_counts
            self.local_parameters[SplinkLocalParameter.ALLELE_COUNT] = allele_counts
            self.set_compensator_flag()

        except Exception as non_missing_count_exception:
            self.log(non_missing_count_exception)
            self.set_operation_status_failed()

    def compute_non_missing_counts(self, start_index, end_index, queue_non_missing, queue_allele_counts):
        """ Compute local non-missing sample count as well as first/second allele count for a sub-chunk """

        # init dictionaries
        non_missing_sample_counts = dict()
        allele_counts = dict()

        try:
            for snp_index in np.arange(start_index, end_index):

                # put results in the queue whenever computation is done for 1000 SNPs
                if snp_index % 1001 == 1000:
                    queue_non_missing.put(non_missing_sample_counts)
                    queue_allele_counts.put(allele_counts)
                    non_missing_sample_counts = dict()
                    allele_counts = dict()

                # non-missing sample count
                x_matrix, y_vector = self.get_x_matrix_y_vector(snp_index)
                non_missing_sample_counts[snp_index] = y_vector.size

                # allele count
                snp_values = self.snp_values[snp_index]
                first_allele_count = int(
                    2 * np.where(np.array(snp_values) == SnpValue.HOMOZYGOTE_00)[0].size +
                    np.where(np.array(snp_values) == SnpValue.HETEROZYGOTE)[0].size
                )

                second_allele_count = int(
                    2 * np.where(np.array(snp_values) == SnpValue.HOMOZYGOTE_11)[0].size +
                    np.where(np.array(snp_values) == SnpValue.HETEROZYGOTE)[0].size
                )

                # to stick with the correct mapping allele_name -> allele count, which is based on the sorted allele names in the server
                if self.first_allele_names[snp_index] < self.second_allele_names[snp_index]:
                    allele_counts[snp_index] = np.array([first_allele_count, second_allele_count])
                else:
                    allele_counts[snp_index] = np.array([second_allele_count, first_allele_count])

            # put remaining results in the corresponding queues
            queue_non_missing.put(non_missing_sample_counts)
            queue_allele_counts.put(allele_counts)

        except Exception as count_exception:
            self.log(count_exception)
            self.set_operation_status_failed()

    # ##### minor_allele step related functions
    def minor_allele_step(self):
        """ Update the SNP values based on the global minor/major allele name """

        try:
            # get global minor allele names
            global_minor_allele_names = self.global_parameters[SplinkGlobalParameter.MINOR_ALLELE_NAME]
            global_major_allele_names = self.global_parameters[SplinkGlobalParameter.MAJOR_ALLELE_NAME]

            for snp_index in global_minor_allele_names.keys():
                # if local minor/major allele is different from the global minor/major allele
                if self.second_allele_names[snp_index] != global_major_allele_names[snp_index]:

                    # swap the local minor and major allele names
                    self.first_allele_names[snp_index] = global_minor_allele_names[snp_index]
                    self.second_allele_names[snp_index] = global_major_allele_names[snp_index]

                    # inverse the mapping of the SNP values 0 -> 2 and 2 -> 0
                    self.snp_values[snp_index] = np.where(self.snp_values[snp_index] == 2, -3,
                                                          self.snp_values[snp_index])
                    self.snp_values[snp_index] = np.where(self.snp_values[snp_index] == 0, 2,
                                                          self.snp_values[snp_index])
                    self.snp_values[snp_index] = np.where(self.snp_values[snp_index] == -3, 0,
                                                          self.snp_values[snp_index])

        except Exception as minor_allele_exception:
            self.log(minor_allele_exception)
            self.set_operation_status_failed()

    # ##### contingency table step related functions
    def contingency_table_step(self):
        """ Compute local contingency table for the chunk"""

        try:
            # get SNP indices for which contingency table should be computed
            self.snp_indices = self.global_parameters[SplinkGlobalParameter.SNP_INDEX]
            self.current_chunk_size = len(self.snp_indices)

            # init contingency tables
            self.contingency_tables = dict()

            # queue
            queue_contingency_tables = multiprocessing.Queue()

            # thread to read from the queues
            thread_read_contingency_tables_queue = threading.Thread(target=self.read_queue_contingency_tables,
                                                                    args=(queue_contingency_tables,))
            thread_read_contingency_tables_queue.daemon = True
            thread_read_contingency_tables_queue.start()

            # processes to compute the local contingency tables for the sub-chunks
            process_list = list()
            for start_index_sub_chunk, end_index_sub_chunk in zip(self.sub_chunk_start_indices, self.sub_chunk_end_indices):
                process = multiprocessing.Process(target=self.compute_contingency_table,
                                                  args=(start_index_sub_chunk, end_index_sub_chunk,
                                                        queue_contingency_tables,))

                process_list.append(process)
                process.daemon = True
                process.start()

            # wait for read thread to be done
            thread_read_contingency_tables_queue.join()

            # close queue
            queue_contingency_tables.close()

            # terminate the processes
            for proc in process_list:
                proc.terminate()

            # convert dictionaries to lists;
            contingency_tables = list()
            for snp_index in sorted(self.snp_indices):
                contingency_tables.append(self.contingency_tables[snp_index])

            # share the noisy contingency tables with the server and noise with compensator
            self.local_parameters[SplinkLocalParameter.CONTINGENCY_TABLE] = contingency_tables
            self.set_compensator_flag()

        except Exception as contingency_table_exception:
            self.log(contingency_table_exception)
            self.set_operation_status_failed()

    # compute contingency table for a set of SNPs
    def compute_contingency_table(self, start_snp_index, end_snp_index, contingency_table_queue):
        """ Compute local contingency table for a sub-chunk """

        contingency_tables = dict()

        for snp_index in np.arange(start_snp_index, end_snp_index):
            if snp_index not in self.snp_indices:
                continue

            # put results in the queue whenever computation is done for 1000 SNPs
            if snp_index % 1001 == 1000:
                contingency_table_queue.put(contingency_tables)
                contingency_tables = dict()

            minor_allele = self.first_allele_names[snp_index]
            major_allele = self.second_allele_names[snp_index]

            # minor-case
            minor_case_count = self.compute_allele_count(snp_index, minor_allele, PhenotypeValue.CASE)

            # major-case
            major_case_count = self.compute_allele_count(snp_index, major_allele, PhenotypeValue.CASE)

            # minor-control
            minor_control_count = self.compute_allele_count(snp_index, minor_allele, PhenotypeValue.CONTROL)

            # major-control
            major_control_count = self.compute_allele_count(snp_index, major_allele, PhenotypeValue.CONTROL)

            # contingency table value: [minor-case, major-case, minor-control, major-control]
            contingency_tables[snp_index] = np.array([minor_case_count,
                                                      major_case_count,
                                                      minor_control_count,
                                                      major_control_count])

        # put the remaining contingency tables into queue
        contingency_table_queue.put(contingency_tables)

    # ##### functions related to the beta step of the linear regression
    def beta_linear_step(self):
        """ Compute X'X and X'Y matrices for the chunk """

        try:
            # set dictionaries to empty at the beginning of the chunk
            self.xt_x_matrix = dict()
            self.xt_y_vector = dict()

            # queues
            queue_xt_x_matrix = multiprocessing.Queue()
            queue_xt_y_vector = multiprocessing.Queue()

            # get SNP indices for which X'X and X'Y should be computed
            self.snp_indices = self.global_parameters[SplinkGlobalParameter.SNP_INDEX]
            self.current_chunk_size = len(self.snp_indices)

            # threads to read from the queues
            xt_x_matrix_read_thread = threading.Thread(target=self.read_queue_xt_x_matrix,
                                                       args=(queue_xt_x_matrix,))
            xt_x_matrix_read_thread.daemon = True
            xt_x_matrix_read_thread.start()

            xt_y_vector_read_thread = threading.Thread(target=self.read_queue_xt_y_vector,
                                                       args=(queue_xt_y_vector, ))
            xt_y_vector_read_thread.daemon = True
            xt_y_vector_read_thread.start()

            # processes to compute local X'X and X'Y for the sub-chunks
            process_list = list()
            for start_index_sub_chunk, end_index_sub_chunk in zip(self.sub_chunk_start_indices, self.sub_chunk_end_indices):
                process = multiprocessing.Process(target=self.compute_beta_linear_parameters,
                                                  args=(start_index_sub_chunk, end_index_sub_chunk,
                                                        queue_xt_x_matrix, queue_xt_y_vector))
                process_list.append(process)
                process.daemon = True
                process.start()

            # wait for read threads to be done
            xt_x_matrix_read_thread.join()
            xt_y_vector_read_thread.join()

            # close queues
            queue_xt_x_matrix.close()
            queue_xt_y_vector.close()

            # terminate the processes
            for proc in process_list:
                proc.terminate()

            # convert dictionaries to lists;
            xt_x_matrix = list()
            xt_y_vector = list()
            for snp_index in sorted(self.snp_indices):
                xt_x_matrix.append(self.xt_x_matrix[snp_index])
                xt_y_vector.append(self.xt_y_vector[snp_index])

            # share noisy local X'X matrix and X'Y vector with the server and noise with compensator
            self.local_parameters[SplinkLocalParameter.XT_X_MATRIX] = xt_x_matrix
            self.local_parameters[SplinkLocalParameter.XT_Y_VECTOR] = xt_y_vector
            self.set_compensator_flag()

        except Exception as beta_linear_exception:
            self.log(beta_linear_exception)
            self.set_operation_status_failed()

    def compute_beta_linear_parameters(self, start_index, end_index, queue_xt_x_matrix, queue_xt_y_vector):
        """ Compute local X'X and X'Y for a sub-chunk """

        xt_x_matrices = dict()
        xt_y_vectors = dict()

        for snp_index in np.arange(start_index, end_index):
            if snp_index not in self.snp_indices:
                continue

            # put results in the queue whenever computation is done for 1000 SNPs
            if snp_index % 1001 == 1000:
                queue_xt_x_matrix.put(xt_x_matrices)
                queue_xt_y_vector.put(xt_y_vectors)
                xt_x_matrices = dict()
                xt_y_vectors = dict()

            x_matrix, y_vector = self.get_x_matrix_y_vector(snp_index)

            xt_x_matrices[snp_index] = np.dot(x_matrix.T, x_matrix)
            xt_y_vectors[snp_index] = np.dot(x_matrix.T, y_vector)

        queue_xt_x_matrix.put(xt_x_matrices)
        queue_xt_y_vector.put(xt_y_vectors)

    # ##### functions related to the std error step of the linear regression algorithm
    def std_error_linear_step(self):
        """ Compute local sum square error values for the chunk """

        try:
            # set sse_values dictionary to empty at the beginning of the chunk
            self.sse_values = dict()

            # queue
            queue_sse = multiprocessing.Queue()

            # thread to read from the queues
            sse_read_thread = threading.Thread(target=self.read_queue_sse, args=(queue_sse,))
            sse_read_thread.daemon = True
            sse_read_thread.start()

            # global beta values
            beta_values = self.global_parameters[SplinkGlobalParameter.BETA]
            self.current_chunk_size = len(beta_values)

            # processes to compute the local SSE values for the sub-chunks
            process_list = list()
            for start_index_sub_chunk, end_index_sub_chunk in zip(self.sub_chunk_start_indices, self.sub_chunk_end_indices):
                process = multiprocessing.Process(target=self.compute_sse_values,
                                                  args=(start_index_sub_chunk, end_index_sub_chunk,
                                                        beta_values, queue_sse))
                process_list.append(process)
                process.daemon = True
                process.start()

            # wait for read thread to be done
            sse_read_thread.join()

            # close queue
            queue_sse.close()

            # terminate the processes
            for proc in process_list:
                proc.terminate()

            # convert dictionary to list
            sse_values = list()
            for snp_index in sorted(beta_values.keys()):
                sse_values.append(self.sse_values[snp_index])

            # python list of scalar values must be converted to a numpy array if compensator flag is set
            sse_values = np.array(sse_values)

            # share noisy local sse values with the server and noise with compensator
            self.local_parameters[SplinkLocalParameter.SSE] = sse_values
            self.set_compensator_flag()

        except Exception as std_error_linear_exception:
            self.log(std_error_linear_exception)
            self.set_operation_status_failed()

    def compute_sse_values(self, start_index, end_index, beta_values, queue_sse):
        """ Compute local sum square error value for a sub-chunk """

        sse_values = dict()

        for snp_index in np.arange(start_index, end_index):
            if snp_index not in beta_values.keys():
                continue

            # put results in the queue whenever computation is done for 1000 SNPs
            if snp_index % 1001 == 1000:
                queue_sse.put(sse_values)
                sse_values = dict()

            # compute sum square error value for the SNP
            x_matrix, y_vector = self.get_x_matrix_y_vector(snp_index)
            beta_vector = beta_values[snp_index].reshape(-1, 1)
            y_predicted = np.dot(x_matrix, beta_vector)
            sse_values[snp_index] = np.sum(np.square(y_vector - y_predicted))

        queue_sse.put(sse_values)

    # ##### logistic regression beta step related functions
    def beta_logistic_step(self):
        """ Compute local gradient and Hessian matrices as well as log likelihood values for the chunk """

        try:
            # set gradient, Hessian, and log likelihood dictionaries to empty at the beginning of the chunk
            self.gradient_vectors = dict()
            self.hessian_matrices = dict()
            self.log_likelihood_values = dict()

            # queues
            queue_gradient = multiprocessing.Queue()
            queue_hessian = multiprocessing.Queue()
            queue_log_likelihood = multiprocessing.Queue()

            # thread to read from the queues
            gradient_read_thread = threading.Thread(target=self.read_queue_gradient, args=(queue_gradient,))
            gradient_read_thread.daemon = True
            gradient_read_thread.start()

            hessian_read_thread = threading.Thread(target=self.read_queue_hessian, args=(queue_hessian,))
            hessian_read_thread.daemon = True
            hessian_read_thread.start()

            log_likelihood_read_thread = threading.Thread(target=self.read_queue_log_likelihood, args=(queue_log_likelihood,))
            log_likelihood_read_thread.daemon = True
            log_likelihood_read_thread.start()

            # global beta values and current beta iteration
            beta_values = self.global_parameters[SplinkGlobalParameter.BETA]
            self.current_chunk_size = len(beta_values)
            self.current_beta_iteration = self.global_parameters[SplinkGlobalParameter.CURRENT_BETA_ITERATION]

            # processes to compute the gradient, Hessian, and log likelihood values for sub-chunks
            process_list = list()
            for start_index_sub_chunk, end_index_sub_chunk in zip(self.sub_chunk_start_indices,
                                                                  self.sub_chunk_end_indices):
                process = multiprocessing.Process(target=self.compute_beta_logistic_parameters,
                                                  args=(start_index_sub_chunk, end_index_sub_chunk,
                                                        beta_values, queue_gradient, queue_hessian, queue_log_likelihood))
                process_list.append(process)
                process.daemon = True
                process.start()

            # wait for read thread to be done
            gradient_read_thread.join()
            hessian_read_thread.join()
            log_likelihood_read_thread.join()

            # close queues
            queue_gradient.close()
            queue_hessian.close()
            queue_log_likelihood.close()

            # terminate the processes
            for proc in process_list:
                proc.terminate()

            # convert dictionary to list
            gradient_vectors = list()
            hessian_matrices = list()
            log_likelihood_values = list()
            for snp_index in sorted(beta_values.keys()):
                gradient_vectors.append(self.gradient_vectors[snp_index])
                hessian_matrices.append(self.hessian_matrices[snp_index])
                log_likelihood_values.append(self.log_likelihood_values[snp_index])

            # python list of scalars must be converted to a numpy array if compensator flag is set
            log_likelihood_values = np.array(log_likelihood_values)

            # share the noisy local gradient, Hessian, and log likelihood values with the server and noise with compensator
            self.local_parameters[SplinkLocalParameter.GRADIENT] = gradient_vectors
            self.local_parameters[SplinkLocalParameter.HESSIAN] = hessian_matrices
            self.local_parameters[SplinkLocalParameter.LOG_LIKELIHOOD] = log_likelihood_values
            self.set_compensator_flag()

        except Exception as beta_logistic_exception:
            self.log(beta_logistic_exception)
            self.set_operation_status_failed()

    def compute_beta_logistic_parameters(self, start_index, end_index, beta_values, queue_gradient, queue_hessian, queue_log_likelihood):
        """ Compute local gradient vector, Hessian matrix, and log likelihood values for a sub-chunk """

        epsilon = np.finfo(float).eps  # to avoid log(0)
        gradient_vectors = dict()
        hessian_matrices = dict()
        log_likelihood_values = dict()

        for snp_index in np.arange(start_index, end_index):
            if snp_index not in beta_values.keys():
                continue

            # put results in the queues whenever computation is done for 1000 SNPs
            if snp_index % 1001 == 1000:
                queue_gradient.put(gradient_vectors)
                queue_hessian.put(hessian_matrices)
                queue_log_likelihood.put(log_likelihood_values)
                gradient_vectors = dict()
                hessian_matrices = dict()
                log_likelihood_values = dict()

            x_matrix, y_vector = self.get_x_matrix_y_vector(snp_index)
            beta_vector = beta_values[snp_index].reshape(-1, 1)

            # gradient
            x_beta_product = np.dot(x_matrix, beta_vector)
            y_predicted = 1 / (1 + np.exp(-x_beta_product))
            gradient_vectors[snp_index] = np.dot(x_matrix.T, (y_vector - y_predicted))

            # hessian matrix
            hessian_matrices[snp_index] = np.dot(np.multiply(x_matrix.T, (y_predicted * (1 - y_predicted)).T), x_matrix)

            # log likelihood
            log_likelihood_values[snp_index] = np.sum(
                y_vector * np.log(y_predicted + epsilon) + (1 - y_vector) * np.log(1 - y_predicted + epsilon))

        queue_gradient.put(gradient_vectors)
        queue_hessian.put(hessian_matrices)
        queue_log_likelihood.put(log_likelihood_values)

    # #####  std error step related functions for logistic regression algorithm
    def std_error_logistic_step(self):
        """ Compute local hessian matrices for the chunk """

        try:
            # set Hessian dictionary to empty at the beginning of the chunk
            self.hessian_matrices = dict()

            # queue
            queue_hessian = multiprocessing.Queue()

            # thread to read from the queue
            hessian_read_thread = threading.Thread(target=self.read_queue_hessian, args=(queue_hessian,))
            hessian_read_thread.daemon = True
            hessian_read_thread.start()

            # global beta values
            beta_values = self.global_parameters[SplinkGlobalParameter.BETA]
            self.current_chunk_size = len(beta_values)

            # processes to compute the local Hessian matrices for the sub-chunks
            process_list = list()
            for start_index_sub_chunk, end_index_sub_chunk in zip(self.sub_chunk_start_indices,
                                                                  self.sub_chunk_end_indices):
                process = multiprocessing.Process(target=self.compute_hessian_matrices,
                                                  args=(start_index_sub_chunk, end_index_sub_chunk, beta_values, queue_hessian,))
                process_list.append(process)
                process.daemon = True
                process.start()

            # wait for read thread to be done
            hessian_read_thread.join()

            # close queues
            queue_hessian.close()

            # terminate the processes
            for proc in process_list:
                proc.terminate()

            # convert dictionary to list
            hessian_matrices = list()
            for snp_index in sorted(beta_values.keys()):
                hessian_matrices.append(self.hessian_matrices[snp_index])

            # share noisy local Hessian matrices with the server and noise with compensator
            self.local_parameters[SplinkLocalParameter.HESSIAN] = hessian_matrices
            self.set_compensator_flag()

        except Exception as std_error_logistic_exception:
            self.log(std_error_logistic_exception)
            self.set_operation_status_failed()

    def compute_hessian_matrices(self, start_index, end_index, beta_values, queue_hessian):
        """ Compute local Hessian matrices for a sub-chunk """

        hessian_matrices = dict()

        for snp_index in np.arange(start_index, end_index):
            if snp_index not in beta_values.keys():
                continue

            # put results in the queues whenever computation is done for 1000 SNPs
            if snp_index % 1001 == 1000:
                queue_hessian.put(hessian_matrices)
                hessian_matrices = dict()

            # Hessian matrix
            x_matrix, y_vector = self.get_x_matrix_y_vector(snp_index)
            beta_vector = beta_values[snp_index].reshape(-1, 1)
            x_beta_product = np.dot(x_matrix, beta_vector)
            y_predicted = 1 / (1 + np.exp(-x_beta_product))
            hessian_matrices[snp_index] = np.dot(np.multiply(x_matrix.T, (y_predicted * (1 - y_predicted)).T), x_matrix)

        queue_hessian.put(hessian_matrices)

    # ##### Queue functions
    def read_queue_non_missing(self, queue_non_missing):
        while len(self.non_missing_sample_counts) < self.current_chunk_size:
            sample_count_non_missing = queue_non_missing.get()
            self.non_missing_sample_counts.update(sample_count_non_missing)

    def read_queue_allele_counts(self, queue_allele_counts):
        while len(self.allele_counts) < self.current_chunk_size:
            count_alleles = queue_allele_counts.get()
            self.allele_counts.update(count_alleles)

    def read_queue_contingency_tables(self, queue_contingency_tables):
        while len(self.contingency_tables) < self.current_chunk_size:
            cont_table = queue_contingency_tables.get()
            self.contingency_tables.update(cont_table)

    def read_queue_xt_x_matrix(self, queue_xt_x_matrix):
        while len(self.xt_x_matrix) < self.current_chunk_size:
            xt_x = queue_xt_x_matrix.get()
            self.xt_x_matrix.update(xt_x)

    def read_queue_xt_y_vector(self, queue_xt_y_vector):
        while len(self.xt_y_vector) < self.current_chunk_size:
            xt_y = queue_xt_y_vector.get()
            self.xt_y_vector.update(xt_y)

    def read_queue_sse(self, queue_sse):
        while len(self.sse_values) < self.current_chunk_size:
            sse = queue_sse.get()
            self.sse_values.update(sse)

    def read_queue_gradient(self, queue_gradient):
        while len(self.gradient_vectors) < self.current_chunk_size:
            gradient = queue_gradient.get()
            self.gradient_vectors.update(gradient)

    def read_queue_hessian(self, queue_hessian):
        while len(self.hessian_matrices) < self.current_chunk_size:
            hessian_matrix = queue_hessian.get()
            self.hessian_matrices.update(hessian_matrix)

    def read_queue_log_likelihood(self, queue_log_likelihood):
        while len(self.log_likelihood_values) < self.current_chunk_size:
            log_likelihood = queue_log_likelihood.get()
            self.log_likelihood_values.update(log_likelihood)

    # ##### multi-processing functions
    def set_sub_chunk_indices(self, start_snp_index, end_snp_index):
        """ Determine start/end indices for sub-chunks assigned to each process/core """

        try:
            if end_snp_index <= start_snp_index:
                self.log("end_snp_index must be greater than start_snp_index!")
                self.set_operation_status_failed()
                return

            # ensure each process/core will compute at least one SNP statistics
            if self.current_chunk_size < self.cpu_cores:
                cpu_cores = 1
            else:
                cpu_cores = self.cpu_cores

            sub_chunk_size = int(np.ceil(self.current_chunk_size / cpu_cores))

            start_indices = np.arange(start_snp_index, end_snp_index, sub_chunk_size)
            end_indices = start_indices + sub_chunk_size
            end_indices[-1] = end_snp_index

            self.sub_chunk_start_indices = start_indices
            self.sub_chunk_end_indices = end_indices

        except Exception as sub_chunk_exception:
            self.log(sub_chunk_exception)
            self.set_operation_status_failed()

    # ##### Helper functions
    def get_x_matrix_y_vector(self, snp_index):
        """ Create feature matrix and label vector """

        try:
            # get non-missing rows after considering SNP values
            snp_indices_non_missing = self.snp_values[snp_index] != MissingValue.SNP
            index_values_non_missing = np.logical_and(self.non_missing_index_values, snp_indices_non_missing)

            # create feature matrix
            snp_vector = self.snp_values[snp_index][index_values_non_missing].reshape(-1, 1).astype(np.int64)

            if len(self.covariate_names) == 0:
                x_matrix = np.concatenate((np.ones((len(snp_vector), 1)).astype(np.int64), snp_vector), axis=1)
            else:
                x_matrix = np.concatenate((np.ones((len(snp_vector), 1)).astype(np.int64),
                                           snp_vector, self.covariate_matrix[index_values_non_missing, :]), axis=1)

            # create label vector
            y_vector = self.phenotype_values[index_values_non_missing].reshape(-1, 1)
            if self.algorithm == SplinkAlgorithm.LOGISTIC_REGRESSION or self.algorithm == SplinkAlgorithm.CHI_SQUARE:
                y_vector = y_vector.astype(np.uint8)

            return x_matrix, y_vector

        except Exception as x_y_exception:
            self.log(f'{x_y_exception}')
            self.set_operation_status_failed()

    def compute_allele_count(self, snp_index, allele_name, trait):
        """ Compute allele count for minor-case, minor-control, major-case, and major-control """

        try:
            x_matrix, phenotype_values = self.get_x_matrix_y_vector(snp_index)
            snp_values = x_matrix[:, 1]

            trait_indices = np.where(phenotype_values == trait)[0]
            trait_snp_values = snp_values[trait_indices]

            if allele_name == self.first_allele_names[snp_index]:
                return int(2 * np.where(trait_snp_values == SnpValue.HOMOZYGOTE_00)[0].size +
                           np.where(trait_snp_values == SnpValue.HETEROZYGOTE)[0].size)

            if allele_name == self.second_allele_names[snp_index]:
                return int(2 * np.where(trait_snp_values == SnpValue.HOMOZYGOTE_11)[0].size +
                           np.where(trait_snp_values == SnpValue.HETEROZYGOTE)[0].size)

        except Exception as allele_count_exception:
            self.log(allele_count_exception)
            self.set_operation_status_failed()

    # ##### Project progress/status widget related functions
    def get_project_step_text(self):
        """ Customize the label shown for project step in the status widget """
        if self.operation_status == ClientOperation.WAITING_FOR_START:
            return '-'

        if self.algorithm == SplinkAlgorithm.LOGISTIC_REGRESSION and self.project_step == SplinkProjectStep.BETA_LOGISTIC:
            return f'Beta ({self.current_beta_iteration}/{self.max_iterations})'

        if self.algorithm == SplinkAlgorithm.LINEAR_REGRESSION and self.project_step == SplinkProjectStep.BETA_LINEAR:
            return 'Beta'

        if self.algorithm == SplinkAlgorithm.LOGISTIC_REGRESSION and self.project_step == SplinkProjectStep.STD_ERROR_LOGISTIC:
            return 'STD-Error'

        if self.algorithm == SplinkAlgorithm.LINEAR_REGRESSION and self.project_step == SplinkProjectStep.STD_ERROR_LINEAR:
            return 'STD-Error'

        return self.project_step

    def get_chunk_text(self):
        """ Customize label shown for chunk in the status widget """

        if self.total_chunks == -1:
            return '-'

        return f'{self.current_chunk}/{self.total_chunks}'

    # ##### Local parameter value computation
    def compute_local_parameters(self):
        """ OVERRIDDEN: Compute the local parameters in each step of the sPLINK algorithms """

        try:

            super().pre_compute_local_parameters()  # MUST be called BEFORE step functions

            # sPLINK specific local parameter computation steps
            if self.project_step == HyFedProjectStep.INIT:
                self.init_step()

            elif self.project_step == SplinkProjectStep.SNP_ID:
                self.snp_id_step()

            elif self.project_step == SplinkProjectStep.ALLELE_NAME:
                self.allele_name_step()

            elif self.project_step == SplinkProjectStep.SAMPLE_COUNT:
                self.sample_count_step()

            elif self.project_step == SplinkProjectStep.NON_MISSING_COUNT:
                self.non_missing_count_step()

            elif self.project_step == SplinkProjectStep.MINOR_ALLELE:
                self.minor_allele_step()

            elif self.project_step == SplinkProjectStep.CONTINGENCY_TABLE:
                self.contingency_table_step()

            elif self.project_step == SplinkProjectStep.BETA_LINEAR:
                self.beta_linear_step()

            elif self.project_step == SplinkProjectStep.BETA_LOGISTIC:
                self.beta_logistic_step()

            elif self.project_step == SplinkProjectStep.STD_ERROR_LINEAR:
                self.std_error_linear_step()

            elif self.project_step == SplinkProjectStep.STD_ERROR_LOGISTIC:
                self.std_error_logistic_step()

            elif self.project_step == HyFedProjectStep.RESULT:
                super().result_step()  # the result step downloads the result file as zip (it is algorithm-agnostic)
            elif self.project_step == HyFedProjectStep.FINISHED:
                super().finished_step()  # The operations in the last step of the project is algorithm-agnostic

            super().post_compute_local_parameters()  # # MUST be called AFTER step functions
        except Exception as computation_exception:
            self.log(computation_exception)
            super().post_compute_local_parameters()
            self.set_operation_status_failed()
