import sys
from pyspark.sql import SparkSession
import argparse
from pyspark.sql.functions import col, when
from pyspark.sql.types import StringType
from pyspark.sql import functions as F

def create_dataframe(filepath, format, spark):
    """
    Create a spark df given a filepath and format.

    :param filepath: <str>, the filepath
    :param format: <str>, the file format (e.g. "csv" or "json")
    :param spark: <str> the spark session

    :return: the spark df uploaded
    """

    if format == "csv":
        spark_df = spark.read.csv(filepath, header=True, inferSchema=True)
    elif format == "json":
        spark_df = spark.read.json(filepath)
    else:
        raise ValueError("Unsupported file format. Supported formats: csv, json")

    return spark_df


def transform_nhis_data(nhis_df):
    """
    Transform df elements

    :param nhis_df: spark df
    :return: spark df, transformed df
    """

    #add your code here

    nhis_df = nhis_df.withColumn("SEX", when(col("SEX").isNull(), 9).otherwise(col("SEX")))

    nhis_df = nhis_df.withColumn("MRACBPI2", when(col("MRACBPI2") == 3, 4)
                                .when(col("MRACBPI2").isin(6, 7, 12), 3)
                                .when(col("MRACBPI2").isin(16, 17), 6)
                                .when(col("HISPAN_I") != 12, 5)
                                .otherwise(col("MRACBPI2")))
    
    nhis_df = nhis_df.withColumn("AGE_P", when(col("AGE_P").between(18, 24), 1)
                                 .when(col("AGE_P").between(25, 29), 2)
                                 .when(col("AGE_P").between(30, 34), 3)
                                 .when(col("AGE_P").between(35, 39), 4)
                                 .when(col("AGE_P").between(40, 44), 5)
                                 .when(col("AGE_P").between(45, 49), 6)
                                 .when(col("AGE_P").between(50, 54), 7)
                                 .when(col("AGE_P").between(55, 59), 8)
                                 .when(col("AGE_P").between(60, 64), 9)
                                 .when(col("AGE_P").between(65, 69), 10)
                                 .when(col("AGE_P").between(70, 74), 11)
                                 .when(col("AGE_P").between(75, 79), 12)
                                 .when(col("AGE_P") >= 80, 13)
                                 .otherwise(14))

    return nhis_df



def calculate_statistics(joined_df):
    """
    Calculate prevalence statistics

    :param joined_df: the joined df

    :return: None
    """
    # Calculate prevalence by race and ethnic background
    race_ethnic_stats = joined_df.groupBy("_IMPRACE").agg(
        F.count("*").alias("Total"),
        F.sum(F.when(joined_df.DIBEV1 == 1, 1).otherwise(0)).alias("Prevalence")
    )

    # Calculate prevalence by gender
    gender_stats = joined_df.groupBy("SEX").agg(
        F.count("*").alias("Total"),
        F.sum(F.when(joined_df.DIBEV1 == 1, 1).otherwise(0)).alias("Prevalence")
    )

    # Calculate prevalence by BRFSS categorical age
    age_stats = joined_df.groupBy("_AGEG5YR").agg(
        F.count("*").alias("Total"),
        F.sum(F.when(joined_df.DIBEV1 == 1, 1).otherwise(0)).alias("Prevalence")
    )

    # Print the statistics
    print("Prevalence by Race and Ethnic Background:")
    race_ethnic_stats.show()

    print("Prevalence by Gender:")
    gender_stats.show()

    print("Prevalence by BRFSS Categorical Age:")
    age_stats.show()


def join_data(brfss_df, nhis_df):
    """
    Join dataframes

    :param brfss_df: spark df
    :param nhis_df: spark df after transformation
    :return: the joined df

    """

    # Alias columns from NHIS
    nhis_df = nhis_df.select([col("SEX").alias('nhis_SEX')] + [col_name for col_name in nhis_df.columns if col_name != "SEX"])

    # Join using all common columns
    joined_df = brfss_df.join(nhis_df, (brfss_df.SEX == nhis_df.nhis_SEX)
                              & (brfss_df._AGEG5YR == nhis_df.AGE_P)
                              & (brfss_df._IMPRACE == nhis_df.MRACBPI2), 'inner')

    # Drop null values
    joined_df = joined_df.dropna()

    joined_df = joined_df.select("SEX","_AGEG5YR", "_IMPRACE", "_LLCPWT", "DIBEV1")

    return joined_df


# python3 p1.py brfss_input.json nhis_input.csv
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('brfss', type=str, default=None, help="brfss filename")
    arg_parser.add_argument('nhis', type=str, default=None, help="nhis filename")
    arg_parser.add_argument('-o', '--output', type=str, default=None, help="output path(optional)")

    #parse args
    args = arg_parser.parse_args()
    if not args.nhis or not args.brfss:
        arg_parser.usage = arg_parser.format_help()
        arg_parser.print_usage()
    else:
        brfss_filename = args.brfss
        nhis_filename = args.nhis

        # Start spark session
        spark = SparkSession.builder.getOrCreate()

        # load dataframes
        brfss_df = create_dataframe(brfss_filename, 'json', spark)
        nhis_df = create_dataframe(nhis_filename, 'csv', spark)

        # Perform mapping on nhis dataframe
        nhis_df = transform_nhis_data(nhis_df)

        # Join brfss and nhis df
        joined_df = join_data(brfss_df, nhis_df)

        # Calculate statistics
        calculate_statistics(joined_df)

        # Save
        if args.output:
            joined_df.write.csv(args.output, mode='overwrite', header=True)


        # Stop spark session
        spark.stop()