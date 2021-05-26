"""
    A class to open and pre-process a GWAS dataset

    Copyright 2021 Reza NasiriGerdeh, Reihaneh TorkzadehMahani, and Julian Matschinske. All Rights Reserved.

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

from hyfed_client.util.status import OperationStatus

import math
import pandas as pd
import numpy as np


class GwasDataset:
    """ Open and pre-process a GWAS dataset """

    def __init__(self, bed_file_path, phenotype_file_path='', covariate_file_path='', phenotype_name='', covariate_names=()):

        self.bed_file_path = bed_file_path  # .bim and .fam files must be in the same directory as the .bed file
        self.phenotype_file_path = phenotype_file_path
        self.covariate_file_path = covariate_file_path
        self.phenotype_name = phenotype_name  # column name in the phenotype file, indicating the phenotype values
        self.covariate_names = covariate_names  # list of covariate names in the covariate file

        # number of samples not seen in the phenotype/covariate file
        self.unseen_phenotype_samples = 0
        self.unseen_covariate_samples = 0

        # to keep track the log or success/failure of file opening
        self.operation_status = OperationStatus.IN_PROGRESS
        self.error_message = ''
        self.dataset_info = ''

        # attributes corresponding to the fam file; re-initialized in the open_fam_dataframe function
        self.family_id_values = tuple()
        self.individual_id_values = tuple()
        self.sex_values = np.array([])
        self.phenotype_values = np.array([])
        self.sample_count = 0

        # attributes associated with the .bim file; re-initialized in the open_bim_dataframe function
        self.snp_id_values = np.array([])
        self.first_allele_names = dict()
        self.second_allele_names = dict()
        self.maf_zero_snps = 0  # number of SNPs with minor allele frequency of zero that will be ignored; re-initialized in open_bim_dataframe function

        # attribute(s) corresponding to .bed file; re-initialized in the open_bed_file function
        self.snp_values = dict()

        # attribute(s) related to the covariate file
        self.covariate_values = dict()

        # attributes to speed up creating the feature matrix
        self.non_missing_index_values = np.array([])  # re-initialized in the init_non_missing_index_values function
        self.covariate_matrix = np.array([])  # re-initialized in the init_covariate_matrix function

    def open_fam_dataframe(self):
        """ Open the fam file to initialize the arrays containing the sex and phenotype values """

        try:
            fam_file_path = self.bed_file_path[:-3] + 'fam'  # e.g. my_gwas_dataset.bed, my_gwas_dataset.fam
            fam_dataframe = pd.read_csv(fam_file_path, header=None, delim_whitespace=True)

            if fam_dataframe.shape[1] != 6:
                self.operation_status = OperationStatus.FAILED
                self.error_message = fam_file_path + " file must have exactly 6 columns!\n"
                self.error_message += "It might be because the delimiter of the file is not space!"
                return

            self.family_id_values = tuple(fam_dataframe.iloc[:, 0])
            self.individual_id_values = tuple(fam_dataframe.iloc[:, 1])
            self.sex_values = np.array(fam_dataframe.iloc[:, 4]).astype(np.int8)
            self.phenotype_values = np.array(fam_dataframe.iloc[:, 5])
            self.sample_count = fam_dataframe.shape[0]

        except Exception as fam_exception:
            self.operation_status = OperationStatus.FAILED
            self.error_message = f"{fam_exception}"

    def open_bim_dataframe(self):
        """ Open the bim file to initialize the SNP IDs, first/second allele names """

        try:
            bim_file_path = self.bed_file_path[:-3] + 'bim'  # e.g. my_gwas_dataset.bed, my_gwas_dataset.bim
            bim_dataframe = pd.read_csv(bim_file_path, header=None, delimiter="\t")

            if bim_dataframe.shape[1] != 6:
                self.operation_status = OperationStatus.FAILED
                self.error_message = bim_file_path + " file must have exactly 6 columns!\n"
                self.error_message += "It might be because the delimiter of the file is not tab!"
                return

            chromosome_numbers = tuple(bim_dataframe.iloc[:, 0])
            snp_names = tuple(bim_dataframe.iloc[:, 1])
            base_pair_positions = tuple(bim_dataframe.iloc[:, 3])

            # initialize SNP IDs
            self.snp_id_values = list()
            for chrom_num, snp_name, bp_pos in zip(chromosome_numbers, snp_names, base_pair_positions):
                self.snp_id_values.append(str(chrom_num) + "\t" + str(snp_name) + "\t" + str(bp_pos))
            self.snp_id_values = np.array(self.snp_id_values, dtype="S")

            # initialize first/second allele names
            self.first_allele_names = dict(zip(self.snp_id_values, bim_dataframe.iloc[:, 4].astype(str)))
            self.second_allele_names = dict(zip(self.snp_id_values, bim_dataframe.iloc[:, 5].astype(str)))

            # get the number of SNPs with minor allele frequency of zero
            self.maf_zero_snps = np.where(np.array(list(self.first_allele_names.values())) == '0')[0].size

        except Exception as bim_exception:
            self.operation_status = OperationStatus.FAILED
            self.error_message = f"{bim_exception}"

    def open_bed_file(self):
        """ Open bed binary file and initialize the snp values """

        try:
            bed_file = open(self.bed_file_path, "rb")
            first_byte, second_byte, third_byte = bed_file.read(3)

            magic_byte_1 = int('01101100', 2)
            magic_byte_2 = int('00011011', 2)

            if not (first_byte == magic_byte_1 and second_byte == magic_byte_2):
                self.operation_status = OperationStatus.FAILED
                self.error_message = self.bed_file_path + " is not a proper bed file!"
                return

            if third_byte != 1:
                self.operation_status = OperationStatus.FAILED
                self.error_message = "bed file must be snp-major!"
                return

            bed_file.seek(3)
            byte_list = np.fromfile(bed_file, dtype=np.uint8)

            bed_file.close()

            # initialize the snp values
            per_snp_byte_count = math.ceil(self.sample_count / 4)

            for snp_index in range(len(self.snp_id_values)):
                byte_start_index = snp_index * per_snp_byte_count

                genotype_byte_list = byte_list[byte_start_index: byte_start_index + per_snp_byte_count]

                unpacked = np.unpackbits(genotype_byte_list, bitorder='little')[:2 * self.sample_count].reshape(
                    (self.sample_count, 2))
                packed = np.packbits(unpacked, axis=1, bitorder='little').reshape((self.sample_count,)).astype(np.int8)

                packed[packed == 1] = -1  # 10
                packed[packed == 2] = 1  # 01
                packed[packed == 0] = 2  # 00
                packed[packed == 3] = 0  # 11

                self.snp_values[self.snp_id_values[snp_index]] = packed

        except Exception as bed_exception:
            self.operation_status = OperationStatus.FAILED
            self.error_message = f"{bed_exception}"

    def open_phenotype_dataframe(self):
        """ Open phenotype file and re-initialize phenotype values """

        try:
            self.unseen_phenotype_samples = 0

            # open phenotype file
            phenotype_dataframe = pd.read_csv(self.phenotype_file_path, delim_whitespace=True)

            # ensure the phenotype file has a column with name self.phenotype_name
            if self.phenotype_name not in phenotype_dataframe.columns.values:
                self.operation_status = OperationStatus.FAILED
                self.error_message = self.phenotype_name + " is not in " + self.phenotype_file_path + " file columns!"
                return

            phenotype_dtype = phenotype_dataframe[self.phenotype_name].dtype

            # make sure the phenotype file has 'FID' and 'IID' column names
            if 'FID' not in phenotype_dataframe.columns.values:
                self.operation_status = OperationStatus.FAILED
                self.error_message = 'FID is not in ' + self.phenotype_file_path + " file columns!"
                return

            if 'IID' not in phenotype_dataframe.columns.values:
                self.operation_status = OperationStatus.FAILED
                self.error_message = 'IID is not in ' + self.phenotype_file_path + " file columns!"
                return

            # open fam file
            fam_file_path = self.bed_file_path[:-3] + 'fam'
            fam_dataframe = pd.read_csv(fam_file_path, header=None, delim_whitespace=True)

            # match and sort phenotype dataframe based on the fam dataframe
            matched_phenotype_dataframe = pd.DataFrame(columns=phenotype_dataframe.columns)

            for _, row in fam_dataframe.iterrows():
                fam_fid = row[0]
                fam_iid = row[1]
                matched_fid_row = phenotype_dataframe.loc[phenotype_dataframe['FID'] == fam_fid]
                matched_fid_iid_row = matched_fid_row.loc[matched_fid_row['IID'] == fam_iid]

                if matched_fid_iid_row.empty:
                    matched_fid_iid_row = phenotype_dataframe.iloc[0].copy()
                    matched_fid_iid_row.iloc[0] = fam_fid
                    matched_fid_iid_row.iloc[1] = fam_iid
                    self.unseen_phenotype_samples += 1

                    for i in range(2, len(matched_fid_iid_row)):
                        matched_fid_iid_row.iloc[i] = MissingValue.PHENOTYPE

                matched_phenotype_dataframe = matched_phenotype_dataframe.append(matched_fid_iid_row, ignore_index=True)

            # re-initialize the phenotype values
            self.phenotype_values = np.array(matched_phenotype_dataframe[self.phenotype_name]).astype(phenotype_dtype)

        except Exception as phenotype_exception:
            self.operation_status = OperationStatus.FAILED
            self.error_message = f"{phenotype_exception}"

    def open_covariate_dataframe(self):
        """ Open the covariate file and initialize covariate values """

        try:
            self.unseen_covariate_samples = 0

            # open covariate dataframe
            covariate_dataframe = pd.read_csv(self.covariate_file_path, delim_whitespace=True)

            for covariate_name in self.covariate_names:
                if covariate_name not in covariate_dataframe.columns.values:
                    self.operation_status = OperationStatus.FAILED
                    self.error_message = covariate_name + " is not in " + self.covariate_file_path + " column list!"
                    return

            # make sure the covariate file has 'FID' and 'IID' column names
            if 'FID' not in covariate_dataframe.columns.values:
                self.operation_status = OperationStatus.FAILED
                self.error_message = 'FID is not in ' + self.covariate_file_path + " file columns!"
                return

            if 'IID' not in covariate_dataframe.columns.values:
                self.operation_status = OperationStatus.FAILED
                self.error_message = 'IID is not in ' + self.covariate_file_path + " file columns!"
                return

            # open fam file
            fam_file_path = self.bed_file_path[:-3] + 'fam'
            fam_dataframe = pd.read_csv(fam_file_path, header=None, delim_whitespace=True)

            # match and sort phenotype dataframe based on the fam dataframe
            matched_covariate_dataframe = pd.DataFrame(columns=covariate_dataframe.columns)

            covariate_dtypes = dict()
            for covariate_name in covariate_dataframe.columns.values:
                covariate_dtypes[covariate_name] = covariate_dataframe[covariate_name].dtype

            for _, row in fam_dataframe.iterrows():
                fam_fid = row[0]
                fam_iid = row[1]
                matched_fid_row = covariate_dataframe.loc[covariate_dataframe['FID'] == fam_fid]
                matched_fid_iid_row = matched_fid_row.loc[matched_fid_row['IID'] == fam_iid]

                if matched_fid_iid_row.empty:
                    matched_fid_iid_row = covariate_dataframe.iloc[0].copy()
                    matched_fid_iid_row.iloc[0] = fam_fid
                    matched_fid_iid_row.iloc[1] = fam_iid
                    self.unseen_covariate_samples += 1

                    for i in range(2, len(matched_fid_iid_row)):
                        matched_fid_iid_row.iloc[i] = MissingValue.COVARIATE

                matched_covariate_dataframe = matched_covariate_dataframe.append(matched_fid_iid_row, ignore_index=True)

            # initialize covariate values dictionary
            for covariate_name in self.covariate_names:
                self.covariate_values[covariate_name] = np.array(matched_covariate_dataframe[covariate_name]).astype(covariate_dtypes[covariate_name])

        except Exception as covariate_exception:
            self.operation_status = OperationStatus.FAILED
            self.error_message = f"{covariate_exception}"

    def open_and_preprocess(self):
        """ Open and pre-process dataset (fam/bim/bed), phenotype, and covariates files """

        # open files
        self.open_fam_dataframe()
        if self.is_operation_failed():
            return

        self.open_bim_dataframe()
        if self.is_operation_failed():
            return

        if self.phenotype_file_path:
            self.open_phenotype_dataframe()
            if self.is_operation_failed():
                return

        if self.covariate_file_path:
            self.open_covariate_dataframe()
            if self.is_operation_failed():
                return

        # sanity checks for phenotype values
        self.binary_phenotype_sanity_check()
        if self.is_operation_failed():
            return

        # map 1/2 encoded binary phenotypes to 0/1
        self.binary_phenotype_map_to_0_1()
        if self.is_operation_failed():
            return

        # init non_missing indices
        self.init_non_missing_index_values()
        if self.is_operation_failed():
            return

        # init covariate matrix
        self.init_covariate_matrix()
        if self.is_operation_failed():
            return

        # open and load snp values
        self.open_bed_file()

        # dataset info
        self.init_dataset_info()

    # ###### sanity checks
    def binary_phenotype_sanity_check(self):
        """ Check whether binary phenotypes are encoded as 1/2 """

        try:

            if self.is_phenotype_binary():

                # check unique phenotype values
                unique_phenotypes = np.unique(self.phenotype_values)

                # phenotype values must not be the same for all samples
                if unique_phenotypes.size == 1:
                    self.operation_status = OperationStatus.FAILED
                    self.error_message = "Only one unique value appeared in the phenotype values!"
                    return

                if MissingValue.PHENOTYPE in unique_phenotypes and unique_phenotypes.size == 2:
                    self.operation_status = OperationStatus.FAILED
                    self.error_message = "Phenotype values are the same for all samples (ignoring missing phenotypes)!"
                    return

                if (MissingValue.PHENOTYPE in unique_phenotypes and unique_phenotypes.size > 3) or \
                   (MissingValue.PHENOTYPE not in unique_phenotypes and unique_phenotypes.size >= 3):
                    self.operation_status = OperationStatus.FAILED
                    self.error_message = "There are more than two unique values in the phenotype list (ignoring missing phenotypes)!"
                    return

                # phenotype values should be encoded as 1/2
                if (1 not in unique_phenotypes) or (2 not in unique_phenotypes):
                    self.operation_status = OperationStatus.FAILED
                    self.error_message = "Phenotype values are not 1/2 encoded!"
                    return
        except Exception as sanity_exception:
            self.operation_status = OperationStatus.FAILED
            self.error_message = f"{sanity_exception}"

    # #### post-processing functions
    def binary_phenotype_map_to_0_1(self):
        """ Map 1/2 encoded phenotypes to 0/1 """

        try:
            if self.is_phenotype_binary():
                self.phenotype_values = np.where(self.phenotype_values == 1, 0, self.phenotype_values)
                self.phenotype_values = np.where(self.phenotype_values == 2, 1, self.phenotype_values)
                self.phenotype_values = self.phenotype_values.astype(np.int8)
        except Exception as map_exception:
            self.operation_status = OperationStatus.FAILED
            self.error_message = f"{map_exception}"

    # #### attribute initializer functions
    def init_non_missing_index_values(self):
        """ Obtain the sample indices where none of the phenotype, sex, and covariate values are missing """

        try:
            phenotype_indices_non_missing = self.phenotype_values != MissingValue.PHENOTYPE

            sex_indices_non_missing = self.sex_values != MissingValue.SEX

            self.non_missing_index_values = np.logical_and(phenotype_indices_non_missing, sex_indices_non_missing)

            for covariate_name in self.covariate_names:
                covariate_indices_non_missing = self.covariate_values[covariate_name] != MissingValue.COVARIATE
                self.non_missing_index_values = np.logical_and(self.non_missing_index_values,
                                                               covariate_indices_non_missing)
        except Exception as init_exception:
            self.operation_status = OperationStatus.FAILED
            self.error_message = f"{init_exception}"

    def init_covariate_matrix(self):
        """ Initialize the covariate matrix, containing the value of the covariates """

        try:
            for covariate_name in self.covariate_names:
                covariate_vector = self.covariate_values[covariate_name].reshape(-1, 1)

                if self.covariate_matrix.size == 0:
                    self.covariate_matrix = covariate_vector
                else:
                    self.covariate_matrix = np.concatenate((self.covariate_matrix, covariate_vector), axis=1)

        except Exception as init_exception:
            self.operation_status = OperationStatus.FAILED
            self.error_message = f"{init_exception}"

    def init_dataset_info(self):
        try:
            self.dataset_info = '####### dataset info\n'

            # SNP statistics
            self.dataset_info += f"{len(self.snp_values)} SNPs loaded from .bim file\n"

            if self.maf_zero_snps != 0:
                self.dataset_info += f"{self.maf_zero_snps} SNPs ignored due to minor allele frequency of zero\n"

            # sample statistics
            male_count = np.where(self.sex_values == 1)[0].size
            female_count = np.where(self.sex_values == 2)[0].size
            ambiguous_count = self.sample_count - male_count - female_count
            self.dataset_info += f"{self.sample_count} people ({male_count} male, {female_count} female, {ambiguous_count} ambiguous)" + \
                                 " loaded from .fam file\n"

            # phenotype statistics
            missing_phenotype_samples = np.where(self.phenotype_values == MissingValue.PHENOTYPE)[0].size - self.unseen_phenotype_samples
            non_missing_phenotype_samples = np.where(self.phenotype_values != MissingValue.PHENOTYPE)[0].size
            all_phenotype_samples = missing_phenotype_samples + non_missing_phenotype_samples
            if self.phenotype_file_path:
                file_type = "phenotype"
            else:
                file_type = ".fam"

            self.dataset_info += f"{all_phenotype_samples} phenotype values ({non_missing_phenotype_samples} non-missing, " + \
                                 f"{missing_phenotype_samples} missing) loaded from {file_type} file\n"

            if self.unseen_phenotype_samples != 0:
                self.dataset_info += f"{self.unseen_phenotype_samples} people not seen in {file_type} file\n"

            # covariate statistics
            if self.covariate_file_path:
                covariate_count = self.sample_count - self.unseen_covariate_samples

                self.dataset_info += f"{covariate_count} values loaded from covariate file for each covariate: "

                for covariate_name in self.covariate_names:
                    missing_covariate_samples = np.where(self.covariate_values[covariate_name] == MissingValue.COVARIATE)[0].size - self.unseen_covariate_samples
                    non_missing_covariate_samples = np.where(self.covariate_values[covariate_name] != MissingValue.COVARIATE)[0].size
                    self.dataset_info += f"'{covariate_name}' ({non_missing_covariate_samples} non-missing, {missing_covariate_samples} missing), "

                self.dataset_info = self.dataset_info[:-2] + "\n"

                if self.unseen_covariate_samples != 0:
                    self.dataset_info += f"{self.unseen_covariate_samples} people not seen in covariate file\n"

            # phenotypes considered/ignored
            phenotypes_considered = np.where(self.non_missing_index_values)[0].size
            phenotype_count_ignored = self.sample_count - phenotypes_considered
            if self.is_phenotype_binary():
                phenotype_list = self.phenotype_values[self.non_missing_index_values]
                case_count = np.where(phenotype_list == PhenotypeValue.CASE)[0].size
                control_count = np.where(phenotype_list == PhenotypeValue.CONTROL)[0].size
                self.dataset_info += f"{phenotypes_considered} phenotypes ({case_count} cases, {control_count} controls) " + \
                                     f"considered and {phenotype_count_ignored} phenotypes ignored\n\n"
            else:
                self.dataset_info += f"{phenotypes_considered} phenotypes considered and " + \
                         f"{phenotype_count_ignored} phenotypes ignored\n\n"

        except Exception as dataset_info_exception:
            self.operation_status = OperationStatus.FAILED
            self.error_message = f"{dataset_info_exception}"

    # ###### getter functions
    def get_sex_values(self):
        return self.sex_values

    def get_phenotype_values(self):
        return self.phenotype_values

    def get_sample_count(self):
        return self.sample_count

    def get_snp_id_values(self):
        return self.snp_id_values

    def get_first_allele_names(self):
        return self.first_allele_names

    def get_second_allele_names(self):
        return self.second_allele_names

    def get_snp_values(self):
        return self.snp_values

    def get_covariate_values(self):
        return self.covariate_values

    def get_non_missing_index_values(self):
        return self.non_missing_index_values

    def get_covariate_matrix(self):
        return self.covariate_matrix

    def get_error_message(self):
        return self.error_message

    def get_dataset_info(self):
        return self.dataset_info

    def is_operation_failed(self):
        return self.operation_status == OperationStatus.FAILED

    def is_phenotype_binary(self):
        dtype = self.phenotype_values.dtype
        if dtype == np.int8 or dtype == np.int16 or dtype == np.int32 or dtype == np.int64:
            return True
        else:
            return False


class MissingValue:
    """ Missing values for snp, phenotype, etc """

    PHENOTYPE = -9
    SEX = -9
    SNP = -1
    COVARIATE = -9
    ALLELE = 0


class SnpValue:
    """ Non-missing snp values for 00, 01/10, and 11 allele combinations """

    HOMOZYGOTE_11 = 0
    HETEROZYGOTE = 1
    HOMOZYGOTE_00 = 2


class PhenotypeValue:
    """ Non-missing binary phenotype values """

    CASE = 1
    CONTROL = 0
