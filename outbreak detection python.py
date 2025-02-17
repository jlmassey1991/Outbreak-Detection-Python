# -*- coding: utf-8 -*-
"""
Outbreak Detection Surveillance Adaptation
Created on Mon Feb 17 12:55:50 2025
"""

from transforms.api import transform_df, Input, Output
from pyspark.sql import functions as F
from myproject.datasets import constants


invalid_foods = constants.INVALID_FOOD_ITEMS


# Clean string values
def clean_data(df):
    """
    Remove empty strings, double quotes (including smart quotes), and trim white spaces from string columns.
    Converts empty strings or strings with only spaces to null.
    """
    # Define a pattern to match all types of double quotes
    double_quotes_pattern = (
        r"[\"\u201C\u201D]"  # Matches standard (") and smart quotes (“ and ”)
    )


    for column_name, dtype in df.dtypes:
        if dtype == "string":
            # Replace empty or whitespace-only strings with None, trim whitespaces, and remove double quotes
            df = df.withColumn(
                column_name,
                F.when(F.trim(F.col(column_name)) == "", None).otherwise(
                    F.regexp_replace(
                        F.trim(F.col(column_name)), double_quotes_pattern, ""
                    )
                ),
            )
    return df




def aggregate_column(
    df,
    group_col,
    agg_col,
    alias,
    dataset_name=None,
    sort=False,
    arr_dis=True,
    delimiter=";",
):
    """
    Aggregates a column by collecting non-null/empty values, optionally sorting,
    and concatenating into a single string. Handles 'unknown' for specific datasets.
    """
    # Sort dataset
    if "CDCID" in df.columns or "OutbreakMainID" in df.columns:
        primary_id_col = "CDCID" if "CDCID" in df.columns else "OutbreakMainID"
        if "Id" in df.columns:
            df = df.orderBy(primary_id_col, "Id")
        else:
            df = df.orderBy(primary_id_col)


    # Collect non-null and non-empty values
    expr = F.collect_list(F.when(F.trim(F.col(agg_col)) != "", F.col(agg_col)))


    # Sorting, if required
    if sort:
        expr = F.sort_array(expr)


    # Handle 'unknown' values for etiology dataset
    if dataset_name == "etiology":
        # Normalize 'unknown' variations to 'unknown' using regex
        expr = F.transform(
            expr,
            lambda x: F.when(
                F.lower(F.trim(x)).rlike(r"^(unknown|unknown\s+.*)$"), "unknown"
            ).otherwise(x),
        )


        # Remove duplicates if arr_dis is True
        if arr_dis:
            expr = F.array_distinct(expr)


        # Check if all values are 'unknown'
        is_all_unknown = (
            (F.size(expr) > 0)
            & (F.size(F.array_distinct(expr)) == 1)
            & F.array_contains(expr, "unknown")
        )


        concatenated = F.when(is_all_unknown, F.lit("unknown")).otherwise(
            F.concat_ws(delimiter, expr)
        )
    else:
        # Non-etiology case: Handle array distinct and concatenation
        expr = F.array_distinct(expr) if arr_dis else expr
        concatenated = F.concat_ws(delimiter, expr)


    # Replace empty results or delimiter-only strings with null
    result = F.when(F.trim(concatenated) == "", F.lit(None)).otherwise(concatenated)


    # Perform the aggregation
    return df.groupBy(group_col).agg(result.alias(alias))




@transform_df(
    Output("ri.foundry.main.dataset.264feeab-2a02-4389-9616-9cfa648fc542"),
    obmain=Input("ri.foundry.main.dataset.390fe829-8b1a-4085-8d7e-603245e9a519"),
    setting=Input("ri.foundry.main.dataset.26e4d9e8-136b-4f0e-8244-336afcfc5cb3"),
    foodvehicle=Input("ri.foundry.main.dataset.0f0913b3-2886-4da6-bda2-f6f5a570b17c"),
    foodingredient=Input(
        "ri.foundry.main.dataset.397d7d72-bc8e-4af9-8544-6125a952813a"
    ),
    catassignment=Input("ri.foundry.main.dataset.e40c0107-47d8-4e88-ba1b-7f79572e20a8"),
    etiology=Input("ri.foundry.main.dataset.259b50df-b420-4283-8401-a45b163c548f"),
    water_exp=Input("ri.foundry.main.dataset.73559d80-b115-4377-9bb0-b1f598049b47"),
    # Old datasets
    # obmain=Input("ri.foundry.main.dataset.7f526c1c-df76-47b3-9a8b-8d08b285b889"),
    # setting=Input("ri.foundry.main.dataset.ef7f5d04-261f-4562-b9c8-7be39d436615"),
    # foodvehicle=Input("ri.foundry.main.dataset.027aaff4-32b8-4ef7-8682-29640387bd77"),
    # foodingredient=Input(
    #     "ri.foundry.main.dataset.dc4b0868-ca47-4450-a7b6-b5c84292917f"
    # ),
    # catassignment=Input("ri.foundry.main.dataset.6f3a724b-9f7a-417e-bb52-ead57e746b0b"),
)
def compute(
    obmain, setting, foodvehicle, foodingredient, catassignment, etiology, water_exp
):
    # Apply the cleaning function to raw data
    setting = clean_data(setting)
    foodvehicle = clean_data(foodvehicle)
    foodingredient = clean_data(foodingredient)
    catassignment = clean_data(catassignment)
    etiology = clean_data(etiology)


    # Setting
    aggregated_setting = aggregate_column(
        setting, "OutbreakMainID", "SettingName", "Setting"
    )
    df = obmain.join(aggregated_setting, on="OutbreakMainID", how="left")


    # Food Vehicle
    aggregated_foodvehicle = aggregate_column(
        foodvehicle, "OutbreakMainID", "FoodName", "FoodVehicle"
    )
    df = df.join(aggregated_foodvehicle, on="OutbreakMainID", how="left")


    # Food Contaminated Ingredient
    aggregated_foodingredient = aggregate_column(
        foodingredient, "OutbreakMainID", "IngredientName", "FoodContaminatedIngredient"
    )
    df = df.join(aggregated_foodingredient, on="OutbreakMainID", how="left")


    # Water Exposure
    aggregated_water_exp = aggregate_column(
        water_exp, "CDCID", "WaterExposure", "WaterExposure", sort=True
    )
    df = df.join(aggregated_water_exp, on="CDCID", how="left")


    # Animal Type
    animaltype = (
        catassignment.filter(F.col("CategoryGroup") == "Animal (Dashboard)")
        .select("CDCID", "LVL1")
        .groupBy("CDCID")
        .agg(
            F.concat_ws(
                ";", F.sort_array(F.array_distinct(F.collect_list("LVL1")))
            ).alias("AnimalType")
        )
    )
    df = df.join(
        animaltype.withColumnRenamed("CDCID", "OutbreakMainID"),
        on="OutbreakMainID",
        how="left",
    )


    # Water Type
    watertype = (
        catassignment.filter(
            (F.col("CategoryGroup") == "Implicated Water System")
            | (F.col("CategoryGroup") == "Venue Type")
        )
        .select("CDCID", "LVL1")
        .groupBy("CDCID")
        .agg(
            F.concat_ws(
                ";", F.sort_array(F.array_distinct(F.collect_list("LVL1")))
            ).alias("WaterType")
        )
    )
    df = df.join(
        watertype.withColumnRenamed("CDCID", "OutbreakMainID"),
        on="OutbreakMainID",
        how="left",
    )


    # Update WaterType column based on PrimaryMode
    df = df.withColumn(
        "WaterType",
        F.when(F.col("PrimaryMode") == "Water", df["WaterType"]).otherwise(None),
    )


    # Etiology
    # Genus and Species
    genus_species = (
        etiology.orderBy("CDCID", "Id")
        .withColumn(
            "GenusSpecies", F.concat_ws(" ", F.col("GenusName"), F.col("SpeciesName"))
        )
        .groupBy("CDCID")
        .agg(F.concat_ws(";", F.collect_list("GenusSpecies")).alias("Etiology"))
        .withColumnRenamed("CDCID", "OutbreakMainID")
    )
    df = df.join(genus_species, on="OutbreakMainID", how="left")


    # Serotype
    # Preprocess the SubtypeName column for the etiology dataset
    etiology = (
        etiology.orderBy("CDCID", "Id")
        .withColumn(
            "ProcessedSubtypeName",
            F.when(
                F.col("GenusName").startswith("Norovirus"),
                F.concat_ws(
                    " ",
                    F.when(
                        F.col("Polymerase").isNotNull(), F.trim(F.col("Polymerase"))
                    ).otherwise(""),
                    F.when(
                        F.col("Capsid").isNotNull(), F.trim(F.col("Capsid"))
                    ).otherwise(""),
                ),
            ).otherwise(
                F.when(
                    F.lower(F.trim(F.col("SubtypeName"))) == "unknown", "unknown"
                ).otherwise(F.col("SubtypeName"))
            ),
        )
        .withColumn(
            "ProcessedSubtypeName",
            F.when(F.trim(F.col("ProcessedSubtypeName")) == "", None).otherwise(
                F.col("ProcessedSubtypeName")
            ),
        )
    )


    serotype = aggregate_column(
        etiology,
        "CDCID",
        "ProcessedSubtypeName",
        "SerotypeOrGenotype",
        dataset_name="etiology",
        arr_dis=False,
    )
    df = df.join(
        serotype.withColumnRenamed("CDCID", "OutbreakMainID"),
        on="OutbreakMainID",
        how="left",
    )


    # Confirmed Status
    confirmed = aggregate_column(
        etiology,
        "CDCID",
        "Confirmed",
        "EtiologyStatus",
        dataset_name="etiology",
        arr_dis=False,
    )


    df = df.join(
        confirmed.withColumnRenamed("CDCID", "OutbreakMainID"),
        on="OutbreakMainID",
        how="left",
    )


    df = df.select(
        "OutbreakMainID",
        "Year",
        "Month",
        "State",
        "PrimaryMode",
        "Etiology",
        "SerotypeOrGenotype",
        "EtiologyStatus",
        "Setting",
        "Illnesses",
        "Hospitalizations",
        "InfoOnHospitalizations",
        "Deaths",
        "InfoOnDeaths",
        "FoodVehicle",
        "FoodContaminatedIngredient",
        "IFSACCategory",
        "WaterExposure",
        "WaterType",
        "AnimalType",
    )


    return df