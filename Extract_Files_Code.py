pip install modis_tools

# Program 2 to download files
from modis_tools.auth import ModisSession
from modis_tools.resources import CollectionApi, GranuleApi
from modis_tools.granule_handler import GranuleHandler

username = "vaishnavi1234"
password = "Vaishnavi@1234"

session = ModisSession(username=username, password=password)

collection_client = CollectionApi(session=session)
collections = collection_client.query(short_name="MODISA_L2_SST", version="R2019.0")

granule_client = GranuleApi.from_collection(collections[0], session=session)

# For Mussafah South Channel
india_bbox = [18.516726, 73.856255, 18.516726, 73.856255]
india_granules = granule_client.query(start_date="2024-04-11", end_date="2024-04-12", bounding_box=india_bbox)

GranuleHandler.download_from_granules(india_granules, session)
