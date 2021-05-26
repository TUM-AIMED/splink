"""
    server-side sPLINK project to aggregate the local parameters from the clients

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

from hyfed_server.project.hyfed_server_project import HyFedServerProject
from hyfed_server.util.hyfed_steps import HyFedProjectStep
from hyfed_server.util.status import ProjectStatus
from hyfed_server.util.utils import client_parameters_to_list

from splink_server.util.splink_steps import SplinkProjectStep
from splink_server.util.splink_parameters import SplinkGlobalParameter, SplinkLocalParameter, SplinkProjectParameter
from splink_server.util.splink_algorithms import SplinkAlgorithm
from splink_server.util.utils import round_result

import io
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import multiprocessing
import threading

import logging
logger = logging.getLogger(__name__)


class SplinkServerProject(HyFedServerProject):
    """ Server side of sPLINK tool """

    def __init__(self, creation_request, project_model):
        """ Initialize sPLINK project attributes based on the values set by the coordinator """

        # initialize base project
        super().__init__(creation_request, project_model)

        try:
            # retrieve the project model instance just created in the parent class
            splink_model_instance = project_model.objects.get(id=self.project_id)

            # chunk_size
            chunk_size = int(creation_request.data[SplinkProjectParameter.CHUNK_SIZE])
            splink_model_instance.chunk_size = chunk_size
            self.chunk_size = chunk_size * 1000

            # covariates
            covariates = creation_request.data[SplinkProjectParameter.COVARIATES]
            splink_model_instance.covariates = covariates
            self.covariates = tuple(
                [covariate_name.strip() for covariate_name in covariates.split(',')]) if covariates else ()

            # max_iterations
            if self.algorithm == SplinkAlgorithm.LOGISTIC_REGRESSION:
                max_iterations = int(creation_request.data[SplinkProjectParameter.MAX_ITERATIONS])
                splink_model_instance.max_iterations = max_iterations
                self.max_iterations = max_iterations

            # result directory
            result_dir = "splink_server/result"
            splink_model_instance.result_dir = result_dir
            self.result_dir = result_dir

            # save the model
            splink_model_instance.save()
            logger.debug(f"Project {self.project_id} : sPLINK specific attributes initialized")

            # #### chunking attributes
            # re-initialized in the init_chunks function
            self.total_chunks = -1
            self.start_indices_chunks = list()
            self.end_indices_chunks = list()

            # re-initialized in the next_chunk functions
            self.current_chunk = 0
            self.chunk_start_index = -1
            self.chunk_end_index = -1
            self.considered_snp_indices = set()  # SNP indices that are NOT ignored due to exceptions
            self.in_process_snp_indices = set()  # SNP indices whose beta values not converged yet (logistic regression)
            self.considered_in_process_snp_indices = set()  # it is always self.considered_snp_indices.intersection(self.in_process_snp_indices)

            # ##### global parameters/values
            self.sample_count = 0
            self.snp_id_values = np.array([])  # SNP IDs common among all clients

            # attributes related to all algorithms
            self.allele_names = dict()  # allele names used as the minor/major allele name across all clients' datasets
            self.non_missing_sample_counts = dict()  # sample count per SNP, where none of phenotype, sex, covariate, and SNP values are missing
            self.allele_counts = dict()  # first/second allele count for each SNP
            self.minor_allele_names = dict()
            self.major_allele_names = dict()
            self.minor_allele_counts = dict()
            self.major_allele_counts = dict()
            self.minor_allele_frequencies = dict()
            self.major_allele_frequencies = dict()
            self.p_values = dict()

            # chi-square algorithm attributes
            self.contingency_tables = dict()
            self.chi_square_values = dict()
            self.odd_ratio_values = dict()
            self.maf_case = dict()  # minor allele frequency for case
            self.maf_control = dict()  # # minor allele frequency for control

            # linear regression attributes
            self.xt_x_matrices = dict()
            self.xt_y_vectors = dict()
            self.xt_x_inverse_matrices = dict()
            self.sse_values = dict()

            # logistic regression attributes
            self.current_beta_iteration = 1
            self.gradient_vectors = dict()
            self.hessian_matrices = dict()
            self.log_likelihood_values = dict()  # re-initialized in minor_allele_step
            self.new_log_likelihood_values = dict()
            self.new_beta_values = dict()  # used in multi-processing
            self.delta_log_likelihood_threshold = 0.0001

            # linear/logistic regression attributes
            self.beta_values = dict()  # re-initialized in minor_allele_step
            self.std_error_values = dict()
            self.t_stat_values = dict()

            # chromosome_number, base_pair_distance, and p-value for ALL chunks
            # used for manhattan plot
            self.chromosome_number_all_chunks = []
            self.base_pair_distance_all_chunks = []
            self.p_value_all_chunks = []

        except Exception as model_exp:
            logger.error(model_exp)
            self.project_failed()

    # ############### Project step functions ####################
    def init_step(self):
        """ Just tell clients to go to the next step """

        try:
            # tell clients to go to the SNP-ID step
            self.set_step(SplinkProjectStep.SNP_ID)

        except Exception as init_exception:
            logger.error(f'Project {self.project_id}: {init_exception}')
            self.project_failed()

    def snp_id_step(self):
        """ Compute the intersection of the SNP IDs from the clients """

        try:
            # compute the SNP IDs common among all clients
            snp_ids_clients = client_parameters_to_list(self.local_parameters, SplinkLocalParameter.SNP_ID)
            intersect_snp_ids = set(snp_ids_clients[0])
            for client_snp_ids in snp_ids_clients:
                intersect_snp_ids = intersect_snp_ids.intersection(client_snp_ids)
            self.snp_id_values = np.array(list(intersect_snp_ids), dtype="S")

            if len(self.snp_id_values) == 0:
                logger.error("There is no SNP common among all clients!")
                self.project_failed()
                return

            # initialize chunks
            self.init_chunks()

            # share common SNP IDs with the clients
            self.global_parameters[SplinkGlobalParameter.SNP_ID] = self.snp_id_values

            # tell clients to go to allele_name step
            self.set_step(SplinkProjectStep.ALLELE_NAME)

        except Exception as snp_id_exception:
            logger.error(f'Project {self.project_id}: {snp_id_exception}')
            self.project_failed()

    def allele_name_step(self):
        """ Ensure the clients employ the same allele name encoding in their datasets """

        try:
            allele_names_clients = client_parameters_to_list(self.local_parameters, SplinkLocalParameter.ALLELE_NAME)

            # initialize allele names for each SNP
            for snp_index in np.arange(len(self.snp_id_values)):
                snp_alleles = list()
                for client_allele_names in allele_names_clients:
                    snp_alleles.append(client_allele_names[0][snp_index])  # first allele name
                    snp_alleles.append(client_allele_names[1][snp_index])  # second allele name

                # allele names must be sorted to make sure the corresponding counts from clients are consistent
                self.allele_names[snp_index] = np.sort(np.unique(snp_alleles))

                # Ensure there are exactly two allele names for the SNP across all clients' datasets
                if len(self.allele_names[snp_index]) != 2:
                    logger.error(f"Project {self.project_id}: clients are using different allele names in their datasets!")
                    self.project_failed()
                    return

            # tell clients to go to sample_count step
            self.set_step(SplinkProjectStep.SAMPLE_COUNT)
        except Exception as allele_name_exception:
            logger.error(f'Project {self.project_id}: {allele_name_exception}')
            self.project_failed()

    def sample_count_step(self):
        """ Compute global sample count and init chunks """

        try:
            # compute global sample count
            sample_counts = client_parameters_to_list(self.local_parameters, SplinkLocalParameter.SAMPLE_COUNT)
            self.sample_count = np.sum(sample_counts)

            # start chunking process
            self.setup_next_chunk()

        except Exception as sample_count_exception:
            logger.error(f'Project {self.project_id}: {sample_count_exception}')
            self.project_failed()

    def non_missing_count(self):
        """ Compute global non-missing sample count as well as allele counts and determine global minor allele """

        try:
            # compute global non-missing sample counts for the SNPs
            clients_non_missing_sample_counts = client_parameters_to_list(self.local_parameters,
                                                                          SplinkLocalParameter.NON_MISSING_SAMPLE_COUNT)
            non_missing_sample_counts = np.sum(clients_non_missing_sample_counts, axis=0)

            # compute global allele counts
            clients_allele_counts = client_parameters_to_list(self.local_parameters, SplinkLocalParameter.ALLELE_COUNT)
            allele_counts = np.sum(clients_allele_counts, axis=0)

            # determine global minor/major allele for each SNP
            minor_allele_names = np.argmin(allele_counts, axis=1)
            major_allele_names = 1 - minor_allele_names

            # get the minor/major allele count for each SNP
            minor_allele_counts = np.min(allele_counts, axis=1)
            major_allele_counts = np.max(allele_counts, axis=1)

            # compute minor/major allele frequency for each SNP
            allele_counts_total = np.sum(allele_counts, axis=1)
            minor_allele_frequencies = minor_allele_counts / allele_counts_total
            major_allele_frequencies = major_allele_counts / allele_counts_total

            # store global non-missing sample count, minor/major allele names/counts/frequencies in the dictionaries;
            snp_counter = -1
            for snp_index in sorted(self.considered_snp_indices.copy()):
                snp_counter += 1
                self.non_missing_sample_counts[snp_index] = non_missing_sample_counts[snp_counter]
                self.allele_counts[snp_index] = allele_counts[snp_counter]
                self.minor_allele_names[snp_index] = self.allele_names[snp_index][minor_allele_names[snp_counter]]
                self.major_allele_names[snp_index] = self.allele_names[snp_index][major_allele_names[snp_counter]]
                self.minor_allele_counts[snp_index] = minor_allele_counts[snp_counter]
                self.major_allele_counts[snp_index] = major_allele_counts[snp_counter]
                self.minor_allele_frequencies[snp_index] = minor_allele_frequencies[snp_counter]
                self.major_allele_frequencies[snp_index] = major_allele_frequencies[snp_counter]

            # share the global minor/major allele names
            minor_allele_names_considered = dict()
            major_allele_names_considered = dict()
            for snp_index in self.considered_snp_indices:
                minor_allele_names_considered[snp_index] = self.minor_allele_names[snp_index]
                major_allele_names_considered[snp_index] = self.major_allele_names[snp_index]

            self.global_parameters[SplinkGlobalParameter.MINOR_ALLELE_NAME] = minor_allele_names_considered
            self.global_parameters[SplinkGlobalParameter.MAJOR_ALLELE_NAME] = major_allele_names_considered

            # tell clients to go to minor-allele step
            self.set_step(SplinkProjectStep.MINOR_ALLELE)

        except Exception as non_missing_count_exception:
            logger.error(f'Project {self.project_id}: {non_missing_count_exception}')
            self.project_failed()

    def minor_allele_step(self):
        """ Determine the next step based on the algorithm name """

        try:
            if self.algorithm == SplinkAlgorithm.CHI_SQUARE:

                # shared SNP indices (excluding ignored ones) for which contingency table should be computed in clients
                self.global_parameters[SplinkGlobalParameter.SNP_INDEX] = self.considered_snp_indices

                # tell clients to go to contingency-table step
                self.set_step(SplinkProjectStep.CONTINGENCY_TABLE)

            elif self.algorithm == SplinkAlgorithm.LINEAR_REGRESSION:

                # shared SNP indices (excluding ignored ones) for which XTX and XTY should be computed in clients
                self.global_parameters[SplinkGlobalParameter.SNP_INDEX] = self.considered_snp_indices

                # tell clients to go to Beta-Linear step
                self.set_step(SplinkProjectStep.BETA_LINEAR)

            elif self.algorithm == SplinkAlgorithm.LOGISTIC_REGRESSION:

                # initialize log_likelihood and beta values
                self.considered_in_process_snp_indices = self.considered_snp_indices.intersection(self.in_process_snp_indices)
                beta_values = dict()  # beta values shared with clients
                for snp_index in self.considered_in_process_snp_indices:
                    # 2 for snp and intercept columns
                    beta_values[snp_index] = np.array([0.0 for _ in range(0, len(self.covariates) + 2)])
                    self.beta_values[snp_index] = beta_values[snp_index]
                    self.log_likelihood_values[snp_index] = None

                # share initial beta values (excluding ignored SNPs) for which gradient, Hessian, and log likelihood
                # should be computed in clients
                self.global_parameters[SplinkGlobalParameter.BETA] = beta_values
                self.global_parameters[SplinkGlobalParameter.CURRENT_BETA_ITERATION] = self.current_beta_iteration

                # tell clients to go to beta-logistic step
                self.set_step(SplinkProjectStep.BETA_LOGISTIC)

        except Exception as minor_allele_exception:
            logger.error(f'Project {self.project_id}: {minor_allele_exception}')
            self.project_failed()

    def contingency_table_step(self):
        """ Compute global chi-square, odd ratio, and p-values using the aggregated contingency tables """

        try:
            clients_contingency_tables = client_parameters_to_list(self.local_parameters, SplinkLocalParameter.CONTINGENCY_TABLE)
            contingency_tables = np.sum(clients_contingency_tables, axis=0)

            # convert global contingency table from list to dictionary
            snp_counter = -1
            for snp_index in sorted(self.considered_snp_indices):
                snp_counter += 1
                self.contingency_tables[snp_index] = contingency_tables[snp_counter]

            # compute the results (i.e. MAF, chi-square, odd-ratio, and p-values) for the chunk
            self.compute_results_chi_square()

            # add chromosome number, base pair distance, and p-value of the current chunk to results for all chunks
            self.append_to_results_all_chunks()

            # save the results using a separate process
            save_process = multiprocessing.Process(target=self.save_results_chi_square)
            save_process.daemon = True
            save_process.start()
            save_process.join()
            save_process.terminate()

            # empty the dictionaries to release the memory because they are not needed anymore
            self.init_algorithm_attributes()

            # if this is not the last chunk, set up the next chunk of SNPs
            if not self.is_last_chunk():
                self.setup_next_chunk()
            else:
                # if this is the last chunk, generate the manhattan plot first, and then, tell clients to download the results
                self.manhattan_plot()
                self.set_step(HyFedProjectStep.RESULT)

        except Exception as contingency_table_exception:
            logger.error(f'Project {self.project_id}: {contingency_table_exception}')
            self.project_failed()

    # ##### linear regression beta-step related functions
    def beta_linear_step(self):
        """ Compute linear regression global beta values using the aggregated XTX and XTY matrices for the chunk """

        try:
            # aggregate X'X matrices and X'Y vectors from the clients
            clients_xt_x_matrices = client_parameters_to_list(self.local_parameters, SplinkLocalParameter.XT_X_MATRIX)
            clients_xt_y_vectors = client_parameters_to_list(self.local_parameters, SplinkLocalParameter.XT_Y_VECTOR)
            xt_x_matrices = np.sum(clients_xt_x_matrices, axis=0)
            xt_y_vectors = np.sum(clients_xt_y_vectors, axis=0)

            # convert lists to dictionaries
            self.xt_x_matrices = dict()
            self.xt_y_vectors = dict()
            snp_counter = -1
            for snp_index in sorted(self.considered_snp_indices.copy()):
                snp_counter += 1
                self.xt_x_matrices[snp_index] = xt_x_matrices[snp_counter]
                self.xt_y_vectors[snp_index] = xt_y_vectors[snp_counter]

            # initialize beta values and xt_x_inverse_matrices as empty dictionaries
            self.beta_values = dict()
            self.xt_x_inverse_matrices = dict()

            # queues
            queue_beta = multiprocessing.Queue()
            queue_xt_x_inverse = multiprocessing.Queue()

            # threads to read from the queue
            beta_read_thread = threading.Thread(target=self.read_queue_beta_linear, args=(queue_beta,))
            beta_read_thread.daemon = True
            beta_read_thread.start()

            xt_x_inverse_read_thread = threading.Thread(target=self.read_queue_xt_x_inverse, args=(queue_xt_x_inverse,))
            xt_x_inverse_read_thread.daemon = True
            xt_x_inverse_read_thread.start()

            # processes to compute the beta values and xt_x_inverse matrices for the sub-chunks
            sub_chunk_start_indices, sub_chunk_end_indices = self.get_start_end_indices(cpu_cores=8)
            process_list = list()
            for start_index_sub_chunk, end_index_sub_chunk in zip(sub_chunk_start_indices, sub_chunk_end_indices):
                process = multiprocessing.Process(target=self.calculate_beta_linear_sub_chunk,
                                                  args=(start_index_sub_chunk, end_index_sub_chunk,
                                                        queue_beta, queue_xt_x_inverse,))
                process_list.append(process)
                process.daemon = True
                process.start()

            # wait for read threads to be done
            beta_read_thread.join()
            xt_x_inverse_read_thread.join()

            # close queues
            queue_beta.close()
            queue_xt_x_inverse.close()

            # terminate the processes
            for proc in process_list:
                proc.terminate()

            # update considered index set
            for snp_index in self.considered_snp_indices:
                if self.beta_values[snp_index][0] == "NA":
                    self.considered_snp_indices.discard(snp_index)
                    self.std_error_values[snp_index] = self.beta_values[snp_index]
                    self.t_stat_values[snp_index] = self.beta_values[snp_index]
                    self.p_values[snp_index] = self.beta_values[snp_index]
                    continue

            # only share beta values for considered SNPs with clients to compute sum square error values
            beta_values = {snp_index: self.beta_values[snp_index] for snp_index in self.considered_snp_indices}
            self.global_parameters[SplinkGlobalParameter.BETA] = beta_values

            # tell clients to go to std-error step
            self.set_step(SplinkProjectStep.STD_ERROR_LINEAR)

        except Exception as beta_linear_exception:
            logger.error(f'Project {self.project_id}: {beta_linear_exception}')
            self.project_failed()

    def calculate_beta_linear_sub_chunk(self, start_index, end_index, queue_beta, queue_xt_x_inverse):
        """ Compute linear regression beta values for a sub-chunk """

        beta_values = dict()
        xt_x_inverse_matrices = dict()

        for snp_index in np.arange(start_index, end_index):
            if snp_index not in self.considered_snp_indices:
                continue

            # put results in the queue whenever computation is done for 1000 SNPs
            if snp_index % 1001 == 1000:
                queue_beta.put(beta_values)
                queue_xt_x_inverse.put(xt_x_inverse_matrices)
                beta_values = dict()
                xt_x_inverse_matrices = dict()

            if np.linalg.det(self.xt_x_matrices[snp_index]) == 0:
                self.beta_values[snp_index] = np.array(["NA" for _ in range(len(self.covariates) + 2)])
                continue

            xt_x_inverse_matrices[snp_index] = np.linalg.inv(self.xt_x_matrices[snp_index])
            beta_values[snp_index] = np.dot(xt_x_inverse_matrices[snp_index], self.xt_y_vectors[snp_index]).flatten()

        queue_beta.put(beta_values)
        queue_xt_x_inverse.put(xt_x_inverse_matrices)

    def read_queue_xt_x_inverse(self, queue_xt_x_inverse):
        while len(self.xt_x_inverse_matrices) < len(self.considered_snp_indices):
            xt_x_inverse = queue_xt_x_inverse.get()
            self.xt_x_inverse_matrices.update(xt_x_inverse)

    def read_queue_beta_linear(self, queue_beta_linear):
        while len(self.beta_values) < len(self.considered_snp_indices):
            betas = queue_beta_linear.get()
            self.beta_values.update(betas)

    # ##### linear regression std-error step related functions
    def std_error_linear_step(self):
        """ Compute linear regression standard error values using the aggregated SSE values """

        try:
            # aggregate SSE values from the clients
            clients_sse_values = client_parameters_to_list(self.local_parameters, SplinkLocalParameter.SSE)
            sse_values = np.sum(clients_sse_values, axis=0)

            # convert sse list to dictionary
            self.sse_values = dict()
            snp_counter = -1
            for snp_index in sorted(self.considered_snp_indices):
                snp_counter += 1
                self.sse_values[snp_index] = sse_values[snp_counter]

            # initialize std_error_values as an empty dictionary
            self.std_error_values = dict()

            # queue
            queue_std_error = multiprocessing.Queue()

            # thread to read from the queue
            std_error_read_thread = threading.Thread(target=self.read_queue_std_error, args=(queue_std_error,))
            std_error_read_thread.daemon = True
            std_error_read_thread.start()

            # processes to compute the std error values for the sub-chunks
            sub_chunk_start_indices, sub_chunk_end_indices = self.get_start_end_indices(cpu_cores=8)
            process_list = list()
            for start_index_sub_chunk, end_index_sub_chunk in zip(sub_chunk_start_indices, sub_chunk_end_indices):
                process = multiprocessing.Process(target=self.calculate_std_error_linear_sub_chunk,
                                                  args=(start_index_sub_chunk, end_index_sub_chunk, queue_std_error,))
                process_list.append(process)
                process.daemon = True
                process.start()

            # wait for read thread to be done
            std_error_read_thread.join()

            # close queues
            queue_std_error.close()

            # terminate the processes
            for proc in process_list:
                proc.terminate()

            # compute results (i.e. t-stats and p-values) for the chunk
            self.compute_results_regression()

            # add chromosome number, base pair distance, and p-value of the current chunk to results for all chunks
            self.append_to_results_all_chunks()

            # save results
            save_process = multiprocessing.Process(target=self.save_results_regression)
            save_process.daemon = True
            save_process.start()
            save_process.join()
            save_process.terminate()

            # empty the dictionaries to release the memory because they are not needed anymore
            self.init_algorithm_attributes()

            # if this is not the last chunk, set up the next chunk of SNPs
            if not self.is_last_chunk():
                self.setup_next_chunk()
            else:
                # if this is the last chunk, generate the manhattan plot first, and then, tell clients to download the results
                self.manhattan_plot()
                self.set_step(HyFedProjectStep.RESULT)

        except Exception as std_error_linear_exception:
            logger.error(f'Project {self.project_id}: {std_error_linear_exception}')
            self.project_failed()

    def calculate_std_error_linear_sub_chunk(self, start_index, end_index, queue_std_error):
        """ Compute linear regression std error values for a sub-chunk """

        std_error_values = dict()

        for snp_index in np.arange(start_index, end_index):
            if snp_index not in self.considered_snp_indices:
                continue

            # put results in the queue whenever computation is done for 1000 SNPs
            if snp_index % 1001 == 1000:
                queue_std_error.put(std_error_values)
                std_error_values = dict()

            sigma_squared = self.sse_values[snp_index] / (self.non_missing_sample_counts[snp_index] - len(self.covariates) - 2)
            variance_values = (sigma_squared * self.xt_x_inverse_matrices[snp_index]).diagonal()
            std_error_values[snp_index] = np.sqrt(variance_values)

        queue_std_error.put(std_error_values)

    # used in std-error step of linear/logistic regression
    def read_queue_std_error(self, queue_std_error):
        while len(self.std_error_values) < len(self.considered_snp_indices):
            std_error = queue_std_error.get()
            self.std_error_values.update(std_error)

    # ##### logistic regression beta step related functions
    def beta_logistic_step(self):
        """ Compute logistic regression global beta values using the aggregated gradient and Hessian matrices for the chunk """

        try:
            # aggregate gradient, Hessian, and log likelihood values from the clients
            clients_gradient_vectors = client_parameters_to_list(self.local_parameters, SplinkLocalParameter.GRADIENT)
            clients_hessian_matrices = client_parameters_to_list(self.local_parameters, SplinkLocalParameter.HESSIAN)
            clients_log_likelihood_values = client_parameters_to_list(self.local_parameters, SplinkLocalParameter.LOG_LIKELIHOOD)

            gradient_vectors = np.sum(clients_gradient_vectors, axis=0)
            hessian_matrices = np.sum(clients_hessian_matrices, axis=0)
            log_likelihood_values = np.sum(clients_log_likelihood_values, axis=0)

            # convert lists to dictionaries
            self.gradient_vectors = dict()
            self.hessian_matrices = dict()
            self.new_log_likelihood_values = dict()
            snp_counter = -1
            for snp_index in sorted(self.considered_in_process_snp_indices):
                snp_counter += 1
                self.gradient_vectors[snp_index] = gradient_vectors[snp_counter]
                self.hessian_matrices[snp_index] = hessian_matrices[snp_counter]
                self.new_log_likelihood_values[snp_index] = log_likelihood_values[snp_counter]

            # initialize new beta values as an empty dictionary
            self.new_beta_values = dict()

            # queue
            queue_beta_values = multiprocessing.Queue()

            # thread to read from the queue
            beta_value_read_thread = threading.Thread(target=self.read_queue_beta_logistic, args=(queue_beta_values,))
            beta_value_read_thread.daemon = True
            beta_value_read_thread.start()

            # processes to compute the new beta values for the sub-chunks
            sub_chunk_start_indices, sub_chunk_end_indices = self.get_start_end_indices(cpu_cores=8)
            process_list = list()
            for start_index_sub_chunk, end_index_sub_chunk in zip(sub_chunk_start_indices, sub_chunk_end_indices):
                process = multiprocessing.Process(target=self.calculate_beta_logistic_sub_chunk,
                                                  args=(start_index_sub_chunk, end_index_sub_chunk, queue_beta_values,))
                process_list.append(process)
                process.daemon = True
                process.start()

            # wait for read thread to be done
            beta_value_read_thread.join()

            # close queues
            queue_beta_values.close()

            # terminate the processes
            for proc in process_list:
                proc.terminate()

            # update beta values
            for snp_index in self.new_beta_values.keys():
                self.beta_values[snp_index] = self.new_beta_values[snp_index]

            # update considered index set
            for snp_index in self.considered_in_process_snp_indices:
                if self.beta_values[snp_index][0] == "NA":
                    self.considered_snp_indices.discard(snp_index)
                    self.std_error_values[snp_index] = self.beta_values[snp_index]
                    self.t_stat_values[snp_index] = self.beta_values[snp_index]
                    self.p_values[snp_index] = self.beta_values[snp_index]
                    continue

            # check whether beta values for the SNP converged. If so, remove the SNP index from the in_process indices
            for snp_index in self.considered_in_process_snp_indices:
                old_log_likelihood = self.log_likelihood_values[snp_index]
                new_log_likelihood = self.new_log_likelihood_values[snp_index]
                if self.has_converged(old_log_likelihood, new_log_likelihood):
                    self.in_process_snp_indices.discard(snp_index)

            # update log likelihood values
            for snp_index in self.new_log_likelihood_values.keys():
                self.log_likelihood_values[snp_index] = self.new_log_likelihood_values[snp_index]

            # if there are still SNPs whose beta values not converged and max iterations not reached yet,
            # share updated global beta values (excluding those ignored or converged) with the clients and stay in beta_logistic step
            self.considered_in_process_snp_indices = self.considered_snp_indices.intersection(self.in_process_snp_indices)
            if self.current_beta_iteration != self.max_iterations and len(self.considered_in_process_snp_indices) != 0:
                self.current_beta_iteration += 1
                beta_values = {snp_index: self.beta_values[snp_index] for snp_index in self.considered_in_process_snp_indices}
                self.global_parameters[SplinkGlobalParameter.BETA] = beta_values
                self.global_parameters[SplinkGlobalParameter.CURRENT_BETA_ITERATION] = self.current_beta_iteration
                logger.debug(f'Project {self.project_id}: Beta iteration # {self.current_beta_iteration} done!')

            # if beta max iterations reached or all beta values converged, share updated beta values (excluding ignored SNPs)
            # with clients and go to the std_error_logistic step
            else:
                beta_values = {snp_index: self.beta_values[snp_index] for snp_index in self.considered_snp_indices}
                self.global_parameters[SplinkGlobalParameter.BETA] = beta_values
                self.set_step(SplinkProjectStep.STD_ERROR_LOGISTIC)

        except Exception as beta_logistic_exception:
            logger.error(f'Project {self.project_id}: {beta_logistic_exception}')
            self.project_failed()

    def calculate_beta_logistic_sub_chunk(self, start_index, end_index, queue_beta_values):
        """ Compute logistic regression beta values for a sub-chunk """

        new_beta_values = dict()

        for snp_index in np.arange(start_index, end_index):
            if snp_index not in self.considered_in_process_snp_indices:
                continue

            # put results in the queue whenever computation is done for 1000 SNPs
            if snp_index % 1001 == 1000:
                queue_beta_values.put(new_beta_values)
                new_beta_values = dict()

            if np.linalg.det(self.hessian_matrices[snp_index]) == 0:
                new_beta_values[snp_index] = np.array(["NA" for _ in range(len(self.covariates) + 2)])
                continue

            hessian_inverse_matrix = np.linalg.inv(self.hessian_matrices[snp_index])
            beta_update_vector = np.dot(hessian_inverse_matrix, self.gradient_vectors[snp_index])
            new_beta_vector = self.beta_values[snp_index].reshape(-1, 1) + beta_update_vector
            new_beta_values[snp_index] = new_beta_vector.flatten()

        queue_beta_values.put(new_beta_values)

    def read_queue_beta_logistic(self, queue_beta_values):
        while len(self.new_beta_values) < len(self.considered_in_process_snp_indices):
            new_betas = queue_beta_values.get()
            self.new_beta_values.update(new_betas)

    # ##### logistic regression std-error step related functions
    def std_error_logistic_step(self):
        """ Compute logistic regression standard error values using the aggregated Hessian matrices for the chunk """

        try:
            # aggregate Hessian matrices from the clients
            clients_hessian_matrices = client_parameters_to_list(self.local_parameters, SplinkLocalParameter.HESSIAN)
            hessian_matrices = np.sum(clients_hessian_matrices, axis=0)

            # convert list to dictionary
            self.hessian_matrices = dict()
            snp_counter = -1
            for snp_index in sorted(self.considered_snp_indices):
                snp_counter += 1
                self.hessian_matrices[snp_index] = hessian_matrices[snp_counter]

            # initialize std_error_values as an empty dictionary
            self.std_error_values = dict()

            # queue
            queue_std_error = multiprocessing.Queue()

            # thread to read from the queue
            std_error_read_thread = threading.Thread(target=self.read_queue_std_error, args=(queue_std_error,))
            std_error_read_thread.daemon = True
            std_error_read_thread.start()

            # processes to compute the std error values for the sub-chunks
            sub_chunk_start_indices, sub_chunk_end_indices = self.get_start_end_indices(cpu_cores=8)
            process_list = list()
            for start_index_sub_chunk, end_index_sub_chunk in zip(sub_chunk_start_indices, sub_chunk_end_indices):
                process = multiprocessing.Process(target=self.calculate_std_error_logistic_sub_chunk,
                                                  args=(start_index_sub_chunk, end_index_sub_chunk, queue_std_error,))
                process_list.append(process)
                process.daemon = True
                process.start()

            # wait for read thread to be done
            std_error_read_thread.join()

            # close queues
            queue_std_error.close()

            # terminate the processes
            for proc in process_list:
                proc.terminate()

            # update ignored index set
            for snp_index in self.considered_snp_indices:
                if self.std_error_values[snp_index][0] == "NA":
                    self.considered_snp_indices.discard(snp_index)
                    self.t_stat_values[snp_index] = self.std_error_values[snp_index]
                    self.p_values[snp_index] = self.std_error_values[snp_index]
                    continue

            # compute the results (i.e. t-stats and p-values) for the chunk
            self.compute_results_regression()

            # add chromosome number, base pair distance, and p-value of the current chunk to results for all chunks
            self.append_to_results_all_chunks()

            # save results
            save_process = multiprocessing.Process(target=self.save_results_regression)
            save_process.daemon = True
            save_process.start()
            save_process.join()
            save_process.terminate()

            # empty the dictionaries to release the memory because they are not needed anymore
            self.init_algorithm_attributes()

            # if this is not the last chunk, set up the next chunk of SNPs
            if not self.is_last_chunk():
                self.setup_next_chunk()
            else:
                # if this is the last chunk, generate the manhattan plot first, and then, tell clients to download the results
                self.manhattan_plot()
                self.set_step(HyFedProjectStep.RESULT)

        except Exception as std_error_logistic_exception:
            logger.error(f'Project {self.project_id}: {std_error_logistic_exception}')
            self.project_failed()

    def calculate_std_error_logistic_sub_chunk(self, start_index, end_index, queue_std_error):
        """ Compute logistic regression std error values for a sub-chunk """

        std_error_values = dict()

        for snp_index in np.arange(start_index, end_index):
            if snp_index not in self.considered_snp_indices:
                continue

            # put results in the queue whenever computation is done for 1000 SNPs
            if snp_index % 1001 == 1000:
                queue_std_error.put(std_error_values)
                std_error_values = dict()

            if np.linalg.det(self.hessian_matrices[snp_index]) == 0:
                std_error_values[snp_index] = np.array(["NA" for _ in range(len(self.covariates) + 2)])
                continue

            std_error_values[snp_index] = np.sqrt(np.linalg.inv(self.hessian_matrices[snp_index]).diagonal())

        queue_std_error.put(std_error_values)

    # ###############  functions related to all algorithms
    def init_algorithm_attributes(self):
        """ Set the chi-square or linear/logistic regression algorithm related dictionaries to empty """

        self.non_missing_sample_counts = dict()
        self.allele_counts = dict()
        self.minor_allele_names = dict()
        self.major_allele_names = dict()
        self.minor_allele_counts = dict()
        self.major_allele_counts = dict()
        self.minor_allele_frequencies = dict()
        self.major_allele_frequencies = dict()

        self.contingency_tables = dict()
        self.maf_case = dict()
        self.maf_control = dict()
        self.chi_square_values = dict()
        self.odd_ratio_values = dict()

        self.xt_x_matrices = dict()
        self.xt_y_vectors = dict()
        self.xt_x_inverse_matrices = dict()
        self.sse_values = dict()

        self.gradient_vectors = dict()
        self.hessian_matrices = dict()
        self.new_log_likelihood_values = dict()
        self.new_beta_values = dict()
        self.log_likelihood_values = dict()

        self.beta_values = dict()
        self.std_error_values = dict()
        self.t_stat_values = dict()
        self.p_values = dict()

    def compute_p_values(self):
        """ Compute p-values for a chunk with multi-processing """

        try:
            queue_p_values = multiprocessing.Queue()

            # thread to read from the queue
            p_value_read_thread = threading.Thread(target=self.read_queue_p_values, args=(queue_p_values,))
            p_value_read_thread.daemon = True
            p_value_read_thread.start()

            # processes to compute the p-values for the sub-chunks
            sub_chunk_start_indices, sub_chunk_end_indices = self.get_start_end_indices(cpu_cores=8)
            process_list = list()
            for start_index_sub_chunk, end_index_sub_chunk in zip(sub_chunk_start_indices,sub_chunk_end_indices):
                process = multiprocessing.Process(target=self.calculate_p_values_sub_chunk,
                                                  args=(start_index_sub_chunk, end_index_sub_chunk, queue_p_values,))
                process_list.append(process)
                process.daemon = True
                process.start()

            # wait for read thread to be done
            p_value_read_thread.join()

            # close queues
            queue_p_values.close()

            # terminate the processes
            for proc in process_list:
                proc.terminate()

            logger.info(f"Project {self.project_id}: p-value computation is done for chunk # {self.current_chunk}!")

        except Exception as p_value_exception:
            logger.error(f'Project {self.project_id}: {p_value_exception}')
            self.project_failed()

    def calculate_p_values_sub_chunk(self, start_index, end_index, queue_p_values):
        """ Compute p-values for a sub-chunk """

        p_values = dict()

        for snp_index in np.arange(start_index, end_index):
            if snp_index not in self.considered_snp_indices:
                continue

            # put results in the queue whenever computation is done for 1000 SNPs
            if snp_index % 1001 == 1000:
                queue_p_values.put(p_values)
                p_values = dict()

            if self.algorithm == SplinkAlgorithm.CHI_SQUARE:
                p_values[snp_index] = 1 - stats.chi2.cdf(self.chi_square_values[snp_index], 1)
            elif self.algorithm == SplinkAlgorithm.LINEAR_REGRESSION:
                degree_of_freedom = self.non_missing_sample_counts[snp_index] - len(self.covariates) - 2
                p_values[snp_index] = 2 * (1 - stats.t.cdf(np.abs(self.t_stat_values[snp_index]), degree_of_freedom))
            elif self.algorithm == SplinkAlgorithm.LOGISTIC_REGRESSION:
                p_values[snp_index] = 1 - stats.chi2.cdf(np.square(np.array(self.t_stat_values[snp_index])), 1)

        queue_p_values.put(p_values)

    def read_queue_p_values(self, queue_p_values):
        while len(self.p_values) < len(self.considered_snp_indices):
            prob_values = queue_p_values.get()
            self.p_values.update(prob_values)

    # ##### Chi-square result computation/saving functions
    def compute_maf(self):
        """ Compute minor allele frequency of case/control for the chunk """

        try:
            for snp_index in self.considered_snp_indices:
                minor_case = self.contingency_tables[snp_index][0]
                major_case = self.contingency_tables[snp_index][1]
                minor_control = self.contingency_tables[snp_index][2]
                major_control = self.contingency_tables[snp_index][3]

                self.maf_case[snp_index] = minor_case / (minor_case + major_case)
                self.maf_control[snp_index] = minor_control / (minor_control + major_control)

            logger.info(f'Project {self.project_id}: case/control minor allele frequency computation is done for chunk # {self.current_chunk}!')

        except Exception as maf_exception:
            logger.error(f'Project {self.project_id}: {maf_exception}')
            self.project_failed()

    def compute_chi_square_values(self):
        """ Compute chi-square value for the chunk """

        try:
            for snp_index in self.considered_snp_indices:
                # observed allele counts
                observed_allele_counts = self.contingency_tables[snp_index]

                # expected allele counts
                expected_allele_counts = np.zeros(4)
                case_count = self.contingency_tables[snp_index][0] + self.contingency_tables[snp_index][1]
                control_count = self.contingency_tables[snp_index][2] + self.contingency_tables[snp_index][3]
                minor_count = self.contingency_tables[snp_index][0] + self.contingency_tables[snp_index][2]
                major_count = self.contingency_tables[snp_index][1] + self.contingency_tables[snp_index][3]
                total_count = case_count + control_count

                expected_allele_counts[0] = (case_count * minor_count) / total_count
                expected_allele_counts[1] = (case_count * major_count) / total_count
                expected_allele_counts[2] = (control_count * minor_count) / total_count
                expected_allele_counts[3] = (control_count * major_count) / total_count

                # compute chi-square value
                chi_square = np.sum(np.square(observed_allele_counts - expected_allele_counts) / expected_allele_counts)
                self.chi_square_values[snp_index] = chi_square

            logger.info(f"Project {self.project_id}: chi-square computation is done for chunk # {self.current_chunk}!")

        except Exception as chi_square_exception:
            logger.error(f'Project {self.project_id}: {chi_square_exception}')
            self.project_failed()

    def compute_odd_ratio_values(self):
        """ Compute odd ratio value for the chunk """

        try:
            for snp_index in self.considered_snp_indices:
                minor_case = self.contingency_tables[snp_index][0]
                major_case = self.contingency_tables[snp_index][1]
                minor_control = self.contingency_tables[snp_index][2]
                major_control = self.contingency_tables[snp_index][3]

                if (major_case * minor_control) != 0:
                    self.odd_ratio_values[snp_index] = (minor_case * major_control) / (major_case * minor_control)
                else:
                    self.odd_ratio_values[snp_index] = "NA"

            logger.info(f"Project {self.project_id}: odd-ratio computation is done for chunk # {self.current_chunk}!")

        except Exception as odd_ratio_exception:
            logger.error(f'Project {self.project_id}: {odd_ratio_exception}')
            self.project_failed()

    def compute_results_chi_square(self):
        """ Compute MAF for case/control, chi-square, odd-ratio, and p-values for chi-square algorithm """

        try:
            self.compute_maf()
            self.compute_chi_square_values()
            self.compute_odd_ratio_values()
            self.compute_p_values()
        except Exception as result_computation_error:
            logger.error(f"Chi-square result computation error: {result_computation_error}")
            self.project_failed()

    def save_results_chi_square(self):
        """ Save chi-square algorithm results for the chunk into the file """

        try:
            logger.info(f'Project {self.project_id}: Started saving results for chunk # {self.current_chunk}!')

            # create result directory/file if they do not already exist
            result_dir = self.create_result_dir()
            result_file = open(f'{result_dir}/chi-square-result.csv', 'a')

            # write the result file header in the first chunk
            if self.current_chunk == 1:
                result_file.write('CHR,SNP,BP,A1,F_A,F_U,A2,CHISQ,P,OR')

            for snp_index in np.arange(self.chunk_start_index, self.chunk_end_index):
                snp_id = self.snp_id_values[snp_index].decode('utf-8')
                chromosome_number, snp_name, base_pair_distance = snp_id.split('\t')

                minor_allele = self.minor_allele_names[snp_index]
                major_allele = self.major_allele_names[snp_index]
                maf_case = round_result(self.maf_case[snp_index])
                maf_control = round_result(self.maf_control[snp_index])
                chi_square = round_result(self.chi_square_values[snp_index])
                p_value = round_result(self.p_values[snp_index])
                odd_ratio = round_result(self.odd_ratio_values[snp_index])

                csv_row = f'{chromosome_number},{snp_name},{base_pair_distance},{minor_allele},{maf_case},' \
                          f'{maf_control},{major_allele},{chi_square},{p_value},{odd_ratio}'

                result_file.write("\n" + str(csv_row))

            result_file.close()

            logger.info(f'Project {self.project_id}: Saving results done for chunk # {self.current_chunk}!')

        except Exception as save_exception:
            logger.error(f'Project {self.project_id}: {save_exception}')
            self.project_failed()

    # ###### Linear/logistic regression result computation/saving functions
    def compute_t_stat_values(self):
        """ Compute T statistics for the chunk """

        try:
            for snp_index in self.considered_snp_indices:
                self.t_stat_values[snp_index] = self.beta_values[snp_index] / self.std_error_values[snp_index]

            logger.info(f'Project {self.project_id}: T statistics computation done for chunk # {self.current_chunk}!')

        except Exception as t_stats_exception:
            logger.error(f'Project {self.project_id}: {t_stats_exception}')
            self.project_failed()

    def compute_results_regression(self):
        """ Compute t-stat and p-values for the linear/logistic regression algorithm """

        try:
            self.compute_t_stat_values()
            self.compute_p_values()
        except Exception as result_computation_error:
            logger.error(f"Regression result computation error: {result_computation_error}")
            self.project_failed()

    def save_results_regression(self):
        """ Save the linear/logistic regression results for the chunk into the file """

        try:
            # create result directory/file if they do not already exist
            result_dir = self.create_result_dir()

            if self.algorithm == SplinkAlgorithm.LINEAR_REGRESSION:
                result_file = open(f'{result_dir}/linear-regression-result.csv', 'a')
            else:
                result_file = open(f'{result_dir}/logistic-regression-result.csv', 'a')

            # write the result file header in the first chunk
            if self.current_chunk == 1:
                result_file.write('CHR,SNP,BP,A1,TEST,NMISS,BETA,STAT,P')

            for snp_index in np.arange(self.chunk_start_index, self.chunk_end_index):
                snp_id = self.snp_id_values[snp_index].decode('utf-8')
                chromosome_number, snp_name, base_pair_distance = snp_id.split('\t')

                beta_counter = 1
                minor_allele = self.minor_allele_names[snp_index]
                feature_name = 'ADD'
                non_missing_samples = round_result(self.non_missing_sample_counts[snp_index])

                beta_value = round_result(self.beta_values[snp_index][beta_counter])
                t_stat_value = round_result(self.t_stat_values[snp_index][beta_counter])
                p_value = round_result(self.p_values[snp_index][beta_counter])

                csv_row = f'{chromosome_number},{snp_name},{base_pair_distance},' \
                          f'{minor_allele},{feature_name},{non_missing_samples},' \
                          f'{beta_value},{t_stat_value},{p_value}'

                result_file.write("\n" + str(csv_row))

                for covariate in self.covariates:
                    beta_counter += 1
                    beta_value = round_result(self.beta_values[snp_index][beta_counter])
                    t_stat_value = round_result(self.t_stat_values[snp_index][beta_counter])
                    p_value = round_result(self.p_values[snp_index][beta_counter])

                    csv_row = f'{chromosome_number},{snp_name},{base_pair_distance},' \
                              f'{minor_allele},{covariate},{non_missing_samples},' \
                              f'{beta_value},{t_stat_value},{p_value}'

                    result_file.write("\n" + str(csv_row))

            result_file.close()

            logger.info(f'Project {self.project_id}: Saving results done for chunk # {self.current_chunk}!')
        except Exception as save_regression_results_exception:
            logger.error(f'Project {self.project_id}: {save_regression_results_exception}')
            self.project_failed()

    # ############## Chunking functions
    def init_chunks(self):
        """ Set the total number of chunks and start/end indices of the chunks """

        try:
            self.total_chunks = int(np.ceil(len(self.snp_id_values) / self.chunk_size))
            for split in np.array_split(np.arange(len(self.snp_id_values)), self.total_chunks):
                self.start_indices_chunks.append(split[0])
                self.end_indices_chunks.append(split[-1] + 1)

            logger.debug(f'Project {self.project_id}: Initializing of chunks is done!')

        except Exception as init_chunk_exp:
            logger.error(f'Project {self.project_id}: {init_chunk_exp}')
            self.project_failed()

    def setup_next_chunk(self):
        """ For the next chunk of SNPs:
                set the start/end chunk index, increment chunk number,
                set the chunk related global parameter values, and go to non-missing-count step
        """

        try:
            # set the chunk attribute values
            self.chunk_start_index = self.start_indices_chunks[self.current_chunk]
            self.chunk_end_index = self.end_indices_chunks[self.current_chunk]
            self.current_chunk += 1
            self.considered_snp_indices = set(np.arange(self.chunk_start_index, self.chunk_end_index)).copy()
            self.in_process_snp_indices = set(
                np.arange(self.chunk_start_index, self.chunk_end_index)).copy()  # used in BETA step of logistic reg
            self.current_beta_iteration = 1

            # set the chunk global parameter values
            self.global_parameters[SplinkGlobalParameter.CURRENT_CHUNK] = self.current_chunk
            self.global_parameters[SplinkGlobalParameter.TOTAL_CHUNKS] = self.total_chunks
            self.global_parameters[SplinkGlobalParameter.CHUNK_START_INDEX] = self.chunk_start_index
            self.global_parameters[SplinkGlobalParameter.CHUNK_END_INDEX] = self.chunk_end_index
            self.global_parameters[SplinkGlobalParameter.SNP_INDEX] = self.considered_snp_indices

            # tell clients to compute statistics for the new chunk starting from the non-missing-count step
            self.set_step(SplinkProjectStep.NON_MISSING_COUNT)

            logger.debug(f'Project {self.project_id}: Chunk # {self.current_chunk} initialized!')

        except Exception as next_chunk_exp:
            logger.error(f'Project {self.project_id}: {next_chunk_exp}')
            self.project_failed()

    # ############## Helper functions
    def is_last_chunk(self):
        """ Check whether current chunk is the last one """

        return self.current_chunk == self.total_chunks

    def has_converged(self, old_log_likelihood, new_log_likelihood):
        """ Determine whether beta values has converged based on the old and new values of log likelihood """

        try:
            if old_log_likelihood is None:
                return False

            delta_log_likelihood = np.abs(old_log_likelihood - new_log_likelihood)
            if delta_log_likelihood > self.delta_log_likelihood_threshold:
                return False

            return True
        except Exception as convergence_exception:
            logger.error(f'Project {self.project_id}: {convergence_exception}')
            self.project_failed()

    def get_start_end_indices(self, cpu_cores):
        """ Determine start/end indices for sub-chunks assigned to each process/core """

        try:
            chunk_size = self.chunk_end_index - self.chunk_start_index

            # ensure each process/core will compute at least one SNP statistics
            if chunk_size < cpu_cores:
                cpu_cores = 1

            sub_chunk_size = int(np.ceil(chunk_size / cpu_cores))
            start_indices = np.arange(self.chunk_start_index, self.chunk_end_index, sub_chunk_size)
            end_indices = start_indices + sub_chunk_size
            end_indices[-1] = self.chunk_end_index

            return start_indices, end_indices

        except Exception as sub_chunk_exception:
            logger.error(sub_chunk_exception)
            self.project_failed()
            return [], []

    def append_to_results_all_chunks(self):
        """ Add the chromosome numbers, base pair distances, and p-values of the current chunk to
            the corresponding lists for all chunks """

        for snp_index in np.arange(self.chunk_start_index, self.chunk_end_index):
            snp_id = self.snp_id_values[snp_index].decode('utf-8')
            chromosome_number, _ , base_pair_distance = snp_id.split('\t')
            p_value = round_result(self.p_values[snp_index])

            self.chromosome_number_all_chunks.append(chromosome_number)
            self.base_pair_distance_all_chunks.append(base_pair_distance)
            self.p_value_all_chunks.append(p_value)

        logger.debug(f'Project {self.project_id}: Chunk # {self.current_chunk} CHR/BP/P added to results for all chunks!')

    def manhattan_plot(self):
        """ draw Manhattan plot for p-values after processing of all chunks finished """

        try:
            manhattan_dict = {'CHR': self.chromosome_number_all_chunks,
                              'BP': self.base_pair_distance_all_chunks,
                              'P': self.p_value_all_chunks}

            manhattan_df = pd.DataFrame.from_dict(manhattan_dict)

            manhattan_df.loc[manhattan_df.P == 0.0, 'P'] = np.finfo(float).eps

            manhattan_df['P_LOG10'] = -np.log10(manhattan_df.P)

            manhattan_df.CHR = manhattan_df.CHR.astype('category')
            manhattan_df.CHR = manhattan_df.CHR.cat.set_categories(list(set(manhattan_df.CHR)), ordered=True)
            manhattan_df = manhattan_df.sort_values(['CHR', 'BP'])

            manhattan_df['ind'] = range(len(manhattan_df))
            manhattan_df_grouped = manhattan_df.groupby('CHR')

            fig = plt.figure(figsize=(24, 8), dpi=80)
            ax = fig.add_subplot(111)
            colors = ['blue', 'green', 'purple', 'brown']
            x_labels = []
            x_labels_pos = []
            for num, (name, group) in enumerate(manhattan_df_grouped):
                print(name)
                group.plot(kind='scatter', x='ind', y='P_LOG10', color=colors[num % len(colors)], ax=ax)
                x_labels.append(name)
                x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0]) / 2))

                print(x_labels_pos[-1])
            ax.set_xticks(x_labels_pos)
            ax.set_xticklabels(x_labels)
            ax.set_xlim([0, len(manhattan_df)])
            ax.set_xlabel('Chromosome')
            ax.set_ylabel('-log10(p)')

            result_dir = self.create_result_dir()

            plt.savefig(f'{result_dir}/manhattan-plot.png', format='png')

            logger.debug(f'Project {self.project_id}: Manhattan plot created!')

        except Exception as plot_exp:
            logger.error("Exception in Manhattan plot!")
            logger.error(plot_exp)

    # ##############  sPLINK specific aggregation code
    def aggregate(self):
        """ OVERRIDDEN: perform sPLINK-project specific aggregations """

        # The following four lines MUST always be called before the aggregation starts
        super().pre_aggregate()
        if self.status != ProjectStatus.AGGREGATING:  # if project failed or aborted, skip aggregation
            super().post_aggregate()
            return

        logger.info(f'Project {self.project_id}: ############## aggregate ####### ')
        logger.info(f'Project {self.project_id}: #### step {self.step}')

        if self.step == HyFedProjectStep.INIT:  # The first step name MUST always be HyFedProjectStep.INIT
            self.init_step()

        elif self.step == SplinkProjectStep.SNP_ID:
            self.snp_id_step()

        elif self.step == SplinkProjectStep.ALLELE_NAME:
            self.allele_name_step()

        elif self.step == SplinkProjectStep.SAMPLE_COUNT:
            self.sample_count_step()

        elif self.step == SplinkProjectStep.NON_MISSING_COUNT:
            self.non_missing_count()

        elif self.step == SplinkProjectStep.MINOR_ALLELE:
            self.minor_allele_step()

        elif self.step == SplinkProjectStep.CONTINGENCY_TABLE:
            self.contingency_table_step()

        elif self.step == SplinkProjectStep.BETA_LINEAR:
            self.beta_linear_step()

        elif self.step == SplinkProjectStep.BETA_LOGISTIC:
            self.beta_logistic_step()

        elif self.step == SplinkProjectStep.STD_ERROR_LINEAR:
            self.std_error_linear_step()

        elif self.step == SplinkProjectStep.STD_ERROR_LOGISTIC:
            self.std_error_logistic_step()

        elif self.step == HyFedProjectStep.RESULT:
            super().result_step()

        # The following line MUST be the last function call in the aggregate function
        super().post_aggregate()
