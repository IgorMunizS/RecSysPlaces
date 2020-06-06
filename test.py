# -*- coding: utf-8 -*-
#######################################################################################################
# extracao_mutant.py
#
# AUTOR...: Igor Muniz Soares - imunizs@indracompany.com
#
# DATA....: 22/04/2020
#
# OBJETIVO: Extrair 8 arquivos layouts pré-estabelecidos do Hive para ambiente Mutant
#
#######################################################################################################


from pyspark import SparkContext, SparkConf
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, HiveContext
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType
import os
from datetime import date, timedelta, datetime
import glob
import shutil
import udf_functions


class Mutant():
    def __init__(self):
        # self.layouts = ['etl_cac','etl_ctrl', 'etl_fixa', 'etl_pre',
        #                 'etl_mvl_qld', 'etl_mvnq_qld', 'etl_pos', 'etl_posv']

        # self.layouts = ['etl_cac']
        if os.environ['MUTANT_ENV'] == 'dev':
            self.data_hoje = '2020-03-26'
            self.data_ontem = '2010-11-11'
        else:
            self.data_hoje = date.today().strftime("%Y-%m-%d")
            self.data_ontem = (date.today() - timedelta(1)).strftime("%Y-%m-%d")

        self.spark_session = (SparkSession
                              .builder
                              .appName('mutant-extraction-pipeline')
                              .config("hive.metastore.uris", "thrift://localhost:9083", conf=SparkConf()
                                      .set("yarn.spark.queue", os.environ['MR_QUEUENAME']))
                              .enableHiveSupport()
                              .getOrCreate()
                              )

        self.duracao_chamada_udf = F.udf(udf_functions.duracao_chamada, StringType())
        self.transferencia_udf = F.udf(udf_functions.transferencia, StringType())
        self.cod_transferencia_udf = F.udf(udf_functions.cod_transferencia, StringType())
        self.p_transfer_udf = F.udf(udf_functions.p_transfer, StringType())
        self.cod_cluster_udf = F.udf(udf_functions.cod_cluster, StringType())
        self.segmento_udf = F.udf(udf_functions.segmento, StringType())
        self.cod_hexa_udf = F.udf(udf_functions.cod_hexa, StringType())
        # self.detalhar_navegacao_udf = F.udf(udf_functions.detalhar_navegacao, StringType())
        self.status_servico_udf = F.udf(udf_functions.status_servico, StringType())
        self.hostname_udf = F.udf(udf_functions.hostname, StringType())
        self.ip_origem_udf = F.udf(udf_functions.ip_origem, StringType())
        self.agent_login_udf = F.udf(udf_functions.agent_login, StringType())
        self.classe_linha_udf = F.udf(udf_functions.classe_linha, StringType())
        self.nota_ath_udf = F.udf(udf_functions.nota_ath, StringType())
        self.confirmou_nota_ath_udf = F.udf(udf_functions.confirmou_nota_ath, StringType())
        self.direcao_chamada_udf = F.udf(udf_functions.direcao_chamada, StringType())
        self.ivr_origem_udf = F.udf(udf_functions.ivr_origem, StringType())
        self.origem_udf = F.udf(udf_functions.origem, StringType())
        self.plat_segmento_udf = F.udf(udf_functions.plat_segmento, StringType())
        self.pop_udf = F.udf(udf_functions.pop, StringType())
        self.ramal_udf = F.udf(udf_functions.ramal, StringType())
        self.site_dac_udf = F.udf(udf_functions.site_dac, StringType())
        self.site_fisico_ag_login_udf = F.udf(udf_functions.site_fisico_ag_login, StringType())
        self.vag_temp_udf = F.udf(udf_functions.vag_temp, StringType())
        self.vq_udf = F.udf(udf_functions.vq, StringType())
        self.vq_pop_qualidade_udf = F.udf(udf_functions.vq_pop_qualidade, StringType())
        self.nota_ura_udf = F.udf(udf_functions.nota_ura, StringType())
        self.confirmou_nota_ura_udf = F.udf(udf_functions.confirmou_nota_ura, StringType())

    def consulta_hive(self):
        query = "SELECT *" + " FROM {}.{}".format(os.environ["MUTANT_SCHEMA_HIVE"],
                                                  os.environ["MUTANT_CHAMADA_TABLE_HIVE"])
        query += " WHERE {campo_data} >= '{data_ontem}' AND {campo_data} < '{data_hoje}'".format(campo_data='DT_RFRN',
                                                                                                 data_ontem=self.data_ontem,
                                                                                                 data_hoje=self.data_hoje)
        self.chamada_df = self.spark_session.sql(query)
        self.chamada_df = self.chamada_df.select([F.col(c).alias("chm_" + c) for c in self.chamada_df.columns])

        query = "SELECT *" + " FROM {}.{}".format(os.environ["MUTANT_SCHEMA_HIVE"],
                                                  os.environ["MUTANT_NAVEGACAO_TABLE_HIVE"])
        query += " WHERE {campo_data} >= '{data_ontem}' AND {campo_data} < '{data_hoje}'".format(campo_data='DT_RFRN',
                                                                                                 data_ontem=self.data_ontem,
                                                                                                 data_hoje=self.data_hoje)

        self.navegacao_df = self.spark_session.sql(query)
        self.navegacao_df = self.navegacao_df.select([F.col(c).alias("nvg_" + c) for c in self.navegacao_df.columns])

    def aplica_transformacoes(self, tmp_df):

        tmp_df = tmp_df.select(col("nvg_DH_INCO_CHMA").alias("DATAHORA"), col("chm_DS_MENU").alias("CANAL"),
                               col("chm_NR_TLFN").alias("ANI"), col("nvg_CD_DNIS").alias("DNIS"),
                               col("nvg_NR_ISTA").alias("CELULAR"),
                               self.duracao_chamada_udf("nvg_DH_FIM_CHMA", "nvg_DH_INCO_CHMA").alias("DURACAO"),
                               col("nvg_NR_PRTO").alias("PROTOCOLO"),
                               self.transferencia_udf("chm_NM_PRMT", "chm_VL_PRMT_CHMA").alias("TRANSFERENCIA"),
                               self.cod_transferencia_udf("chm_NM_PRMT", "chm_VL_PRMT_CHMA").alias("COD_TRANSFERENCIA"),
                               self.p_transfer_udf("nvg_NM_PRMT", "nvg_VL_PRMT_NVGO").alias("P_TRANSFER"),
                               self.cod_cluster_udf("nvg_NM_PRMT", "nvg_VL_PRMT_NVGO").alias("COD_CLUSTER"),
                               self.segmento_udf("chm_NM_PRMT", "chm_VL_PRMT_CHMA").alias("SEGMENTO"),
                               self.cod_hexa_udf("nvg_NM_PRMT", "nvg_VL_PRMT_NVGO").alias("COD_HEXA"),
                               col("chm_DS_PRFL").alias("PERFIL"), col("nvg_DS_CLSF").alias("DETALHAR_NAVEGACAO"),
                               self.status_servico_udf("nvg_DS_TIPO_CLSF_MVEL", "nvg_NM_TIPO_ERRO").alias(
                                   "STATUS_SERVICO"),
                               self.hostname_udf("chm_NM_PRMT", "chm_VL_PRMT_CHMA").alias("HOSTNAME"),
                               self.ip_origem_udf("chm_NM_PRMT", "chm_VL_PRMT_CHMA").alias("IP_ORIGEM"),
                               col("chm_CD_UNCO_CHMA").alias("COD_CHAMADA"),
                               col("nvg_DH_NVGO_PLTO").alias("DATA_NAVEGACAO"))

        return tmp_df

    def transforma_tabelas(self):
        tmp_df = self.navegacao_df.join(self.chamada_df,
                                        self.chamada_df.chm_cd_unco_chma == self.navegacao_df.nvg_cd_unco_chma)
        final_df = self.aplica_transformacoes(tmp_df)
        tmp_1 = final_df.select("DATAHORA", "CANAL", "ANI", "DNIS", "CELULAR", "DURACAO",
                                "PROTOCOLO", "TRANSFERENCIA", "COD_TRANSFERENCIA",
                                "P_TRANSFER", "COD_CLUSTER",
                                "SEGMENTO",
                                "COD_HEXA",
                                "PERFIL", "HOSTNAME",
                                "IP_ORIGEM", "COD_CHAMADA")

        tmp_1 = tmp_1.dropDuplicates()
        tmp_2 = final_df.select("DETALHAR_NAVEGACAO", "DATA_NAVEGACAO", "COD_CHAMADA")
        tmp_2 = tmp_2.dropDuplicates(["DATA_NAVEGACAO"])
        tmp_2 = tmp_2.groupby("COD_CHAMADA").agg(
            F.concat_ws("|", F.collect_list(tmp_2.DETALHAR_NAVEGACAO)).alias("DETALHAR_NAVEGACAO"))
        tmp_3 = final_df.select("STATUS_SERVICO", "DATA_NAVEGACAO", "COD_CHAMADA")
        tmp_3 = tmp_3.dropDuplicates(["DATA_NAVEGACAO"])
        tmp_3 = tmp_3.groupby("COD_CHAMADA").agg(
            F.concat_ws("|", F.collect_list(tmp_3.STATUS_SERVICO)).alias("STATUS_SERVICO"))
        final_df = tmp_1.join(tmp_2, ["COD_CHAMADA"]).join(tmp_3, ["COD_CHAMADA"])
        final_df = final_df.groupBy(final_df['COD_CHAMADA']).agg(
            F.first(final_df['DATAHORA']).alias("DATAHORA"), F.first(final_df['CANAL']).alias("CANAL"),
            F.first(final_df['ANI']).alias("ANI"), F.first(final_df["DNIS"]).alias("DNIS"),
            F.first(final_df['CELULAR']).alias("CELULAR"), F.first(final_df["DURACAO"]).alias("DURACAO"),
            F.first(final_df['PROTOCOLO']).alias("PROTOCOLO"),
            F.first(final_df["TRANSFERENCIA"], ignorenulls=True).alias("TRANSFERENCIA"),
            F.first(final_df["COD_TRANSFERENCIA"], ignorenulls=True).alias("COD_TRANSFERENCIA"),
            F.first(final_df["P_TRANSFER"], ignorenulls=True).alias("P_TRANSFER"),
            F.first(final_df["COD_CLUSTER"],ignorenulls=True).alias("COD_CLUSTER"),
            F.first(final_df["SEGMENTO"], ignorenulls=True).alias("SEGMENTO"),
            F.first(final_df["COD_HEXA"], ignorenulls=True).alias("COD_HEXA"),
            F.first(final_df["PERFIL"], ignorenulls=True).alias("PERFIL"),
            F.first(final_df["HOSTNAME"], ignorenulls=True).alias("HOSTNAME"),
            F.first(final_df["IP_ORIGEM"], ignorenulls=True).alias("IP_ORIGEM"),
            F.first(final_df["STATUS_SERVICO"]).alias("STATUS_SERVICO"),
            F.first(final_df["DETALHAR_NAVEGACAO"]).alias("DETALHAR_NAVEGACAO")
        )

        return final_df

    def arruma_caminho(self, folder_dest, table):
        dest = folder_dest + "/" + table + "/"
        for files in glob.glob(dest + "*.csv"):
            shutil.move(files, folder_dest + "/" + table + ".csv")
        shutil.rmtree(folder_dest + "/" + table)

    def gera_csv(self, df):
        folder_dest = 'file://{}/{}'.format(os.environ['MUTANT_FOLDER_DEST'], "layout")
        df.coalesce(1).write.option("header", "true").csv(folder_dest, sep=',')
        self.arruma_caminho(os.environ['MUTANT_FOLDER_DEST'], "layout")

    def gera_layouts(self):
        self.consulta_hive()

        final_df = self.transforma_tabelas()
        self.gera_csv(final_df)


mutant = Mutant()
mutant.gera_layouts()
