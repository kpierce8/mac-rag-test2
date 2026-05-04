from geo_pipeline.storage.mongo_client import get_database
db = get_database()                                                                                                                                  
for c in db.chunks.find({'figure_classification.image_type': 'annotated imagery'}):                                                                              
    cls = c['figure_classification']                                                                                                                 
    p = c['provenance']                                                                                                                              
    print(f"{p['filename']}  page {p['page_number']}")                                                                                           
    print(f" {cls['description']}")                                                                                                               
    print(f"  Spatial: {cls.get('spatial_info', 'none')}")                                                                                         
    print() 