def retrieve_data(query: str = None, type: str = "mdx"):
    
    if type == "mdx":
        return retrieve_data_with_mdx(query)
    elif type == "test":
        return retrieve_data_with_test(query)
    else:
        raise ValueError(f"Invalid type: {type}")

def retrieve_data_with_mdx(query: str):
    """
    Retrieve data from the database with metadata
    """
    pass

def retrieve_data_with_test(query: str):
    
    md_table = """
        |   Kalender[Jahr] | Kalender[Monat]   |   [Umsatz SD/CO] |
        |-----------------:|:------------------|-----------------:|
        |             2021 | Apr               |         30264.05 |
        |             2021 | Aug               |          9660.99 |
        |             2021 | Dez               |         31104.06 |
        |             2021 | Feb               |         11619.17 |
        |             2021 | Jan               |         12308.10 |
        |             2021 | Jul               |          4399.77 |
        |             2021 | Jun               |         12705.90 |
        |             2021 | Mai               |          4013.68 |
        |             2021 | Mrz               |          6366.93 |
        |             2021 | Nov               |         43170.69 |
        |             2021 | Okt               |         31067.25 |
        |             2021 | Sep               |          8191.73 |
        |             2022 | Apr               |          8222.00 |
        |             2022 | Aug               |        114949.62 |
        |             2022 | Dez               |        107768.10 |
        |             2022 | Feb               |          7901.51 |
        |             2022 | Jan               |          9099.13 |
        |             2022 | Jul               |         49897.36 |
        |             2022 | Jun               |        242067.31 |
        |             2022 | Mai               |         79558.31 |
        |             2022 | Mrz               |         43886.00 |
        |             2022 | Nov               |        191821.13 |
        |             2022 | Okt               |        126704.83 |
        |             2022 | Sep               |        142962.81 |
        |             2023 | Apr               |        136944.76 |
        |             2023 | Aug               |         94834.47 |
        |             2023 | Dez               |           308.46 |
        |             2023 | Feb               |         11985.62 |
        |             2023 | Jan               |         39018.99 |
        |             2023 | Jul               |         56252.40 |
        |             2023 | Jun               |         93584.47 |
        |             2023 | Mai               |         59338.62 |
        |             2023 | Mrz               |         22982.51 |
        |             2023 | Nov               |         81661.45 |
        |             2023 | Okt               |          4556.49 |
        |             2023 | Sep               |         55172.35 |
        |             2024 | Apr               |          4960.46 |
        |             2024 | Aug               |         94434.55 |
        |             2024 | Dez               |          1682.98 |
        |             2024 | Feb               |         24220.52 |
        |             2024 | Jan               |         17836.38 |
        |             2024 | Jul               |          2534.01 |
        |             2024 | Jun               |         22302.27 |
        |             2024 | Mai               |         59299.74 |
        |             2024 | Mrz               |          5705.99 |
        |             2024 | Nov               |         89776.96 |
        |             2024 | Okt               |          2171.52 |
        |             2024 | Sep               |             0.00 |
    """
    return md_table



