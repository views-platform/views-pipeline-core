from views_pipeline_core.data.utils import download_json, convert_json_to_list_of_dicts
from typing import Optional
from pydantic import BaseModel


class Country(BaseModel):
    country_id: int
    name: str
    capname: Optional[str] = None
    caplong: Optional[str] = None
    caplat: Optional[str] = None
    gwcode: int
    gwsyear: int
    gwsmonth: int
    gwsday: int
    gweyear: int
    gwemonth: int
    gweday: int
    isoname: Optional[str] = None
    isonum: int
    isoab: Optional[str] = None
    month_start: int
    month_end: int
    centroidlong: Optional[str] = None
    centroidlat: Optional[str] = None
    in_africa: int
    in_me: int


class CountryData:
    def __init__(self):
        self.__data_url = "https://raw.githubusercontent.com/prio-data/VIEWS_FAO_index/refs/heads/main/data/processed/MatchingTable_VIEWS_Country_IDs.json"
        self.__data = convert_json_to_list_of_dicts(download_json(self.__data_url))
        # print(self._data)
        if isinstance(self.__data, list):
            self.__countries = self.__countries = [
                Country(**country) for country in self.__data
            ]
        else:
            raise ValueError(
                f"json_data must be a list of dictionaries. Found type: {type(self.__data)}"
            )

    def get_country_by_id(self, country_id):
        for country in self.__countries:
            if country.country_id == country_id:
                return country
        return None

    def get_all_countries(self):
        return self.__countries

    def get_country_by_name(self, name):
        for country in self.__countries:
            if country.name.lower() == name.lower():
                return country
        return None


if __name__ == "__main__":
    country_data = CountryData()
    print(country_data.get_country_by_id(32))
    print(country_data.get_country_by_name("Afghanistan"))