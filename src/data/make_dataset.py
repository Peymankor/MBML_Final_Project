"""Fetch and wrangle raw data."""
import click
import logging
import pandas as pd

from typing import List
from pathlib import Path


def fetch_country_cov(country: str, feat: List = "Confirmed"):
    """Fetch COVID19 information `feat` of a `country`."""
    if isinstance(feat, str):
        feat = [feat]
    df = pd.read_csv(
        "https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv",
        parse_dates=["Date"],
    )
    df.Date = pd.to_datetime(df.Date)
    return df.loc[df.Country == country, ["Date"] + feat]


def fetch_country_mob(in_file, country: str):
    """Get mobility data in file `in_file` from a `country`."""
    df = pd.read_csv(in_file, parse_dates=["date"])
    df = (
        df.loc[
            df.country_region == country,
            [
                "date",
                "retail_and_recreation_percent_change_from_baseline",
                "grocery_and_pharmacy_percent_change_from_baseline",
                "parks_percent_change_from_baseline",
                "transit_stations_percent_change_from_baseline",
            ],
        ]
        .rename(columns={"date": "Date"})
        .reset_index(drop=True)
    )
    df.Date = pd.to_datetime(df.Date)
    return df


@click.command()
@click.argument("input_mobility", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--country", type=str, prompt="Country", help="Country to filter from.")
@click.option("--feat", type=str, help="Feature to extract.", default="Confirmed")
def main(input_mobility, output, country, feat="Confirmed"):
    """Fetch and turn raw data into cleaned data.

    From (../raw) into cleaned data ready to be analyzed (in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info("Fetching COVID19 data from from GitHub.")
    df_cov = fetch_country_cov(country, feat)
    logger.info(f"Fetching mobility data from from file {input_mobility}.")
    df_mob = fetch_country_mob(input_mobility, country).iloc[:57, :]
    df = df_cov.merge(df_mob, on="Date")
    try:
        df.iloc[:, 2:] = df.iloc[:, 2:].astype(int)
    except ValueError:
        logger.debug("Mobility Data not be converted to int: NAs received.")
        df.iloc[:, 2:] = df.iloc[:, 2:].astype(float)

    # duplicated dates on mobility, take the daily mean
    df.to_csv(output, index=False)

    logger.info(f"File generated at {output}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
