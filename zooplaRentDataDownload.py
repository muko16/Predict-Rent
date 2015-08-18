import requests
import json
import csv
import os

host = "http://api.zoopla.co.uk/api/v1/"
action = "property_listings.js"

param= {}
param['api_key'] = "hkp7jznhuq4skfb9chsn8sw3"
#param['area'] = "London"
param['postcode'] = "WC"
param['order_by'] = "age"
param['ordering'] = "ascending" 	# "descending" (default) or "ascending".
param['page_size'] = 100			# max:100 default:10
param['page_number'] = 1 	  		# default:1
param['listing_status'] = "rent" 	# "sale" or "rent"
#param['include_sold'] = 1 			# Defaults to 0
param['include_rented'] = 1 		# Defaults to 0

for p in range(1, 3):
	param['postcode'] = "WC" + str(p)
	param['page_number'] = 1 

	file_path = "rentList_" + param['postcode'] + ".csv"
	if not os.path.exists(file_path):
		o = open(file_path, "wb+")
		f = csv.writer(o)
		#f.writerow(["agent_address","agent_logo","agent_name","agent_phone","country","county","description","details_url","displayable_address","first_published_date","floor_plan","image_caption","image_url","last_published_date","latitude","letting_fees","listing_id","listing_status","longitude","num_bathrooms","num_bedrooms","num_floors","num_recepts","outcode","post_town","price","price_change","property_type","rental_prices_accurate","rental_prices_per_month","rental_prices_per_week","short_description","status","street_name","thumbnail_url"])
		f.writerow(["country","county","displayable_address","first_published_date","last_published_date","latitude","listing_id","listing_status","longitude","num_bathrooms","num_bedrooms","num_floors","num_recepts","outcode","post_town","price","price_change","property_type","rental_prices_accurate","rental_prices_per_month","rental_prices_per_week","status","street_name","walkscore","ws_desc"])
		o.close()

	#log_directory = "log_" + param['area']
	log_directory = "log_" + param['postcode']
	if not os.path.exists(log_directory):
	    os.makedirs(log_directory)

	for x in xrange(0, 100):
		# set endpoint
		endpoint = host + action + "?" + '&'.join("%s=%s" % (key,val) for (key,val) in param.iteritems())
		print endpoint

		# download rent data
		r = requests.get(endpoint)
		if(r.status_code != requests.codes.ok):
			print r.content
			print "code is not ok"
			break
		data = json.loads(r.content)
		#print json.dumps(data, indent=4, sort_keys=True)
		
		# logging
		fileName = log_directory + "/rentData_" + str(param['page_number']) + ".txt"
		fo = open(fileName, "wb")
		fo.write( json.dumps(data, indent=4, sort_keys=True) );
		fo.close()

		#
		f = csv.writer(open(file_path, "a"))

		if "listing" not in data:
			print json.dumps(data, indent=4, sort_keys=True)
			break

		if not data["listing"]:
			print json.dumps(data, indent=4, sort_keys=True)
			break		

		for li in data["listing"]:
			#if "letting_fees" in li:
			#	li["letting_fees"] = li["letting_fees"].encode(encoding='UTF-8')
			#FILEDNAMES = ("agent_address","agent_logo","agent_name","agent_phone","country","county","description","details_url","displayable_address","first_published_date","floor_plan","image_caption","image_url","last_published_date","latitude","letting_fees","listing_id","listing_status","longitude","num_bathrooms","num_bedrooms","num_floors","num_recepts","outcode","post_town","price","price_change","property_type","rental_prices_accurate","rental_prices_per_month","rental_prices_per_week","short_description","status","street_name","thumbnail_url")   
			#HEADER     = dict([ (val,val) for val in FILEDNAMES ])  
			
			#writer = csv.DictWriter(f, FILEDNAMES)   
			#writer.writerows(li)  

			#f.writerow(["agent_address","agent_logo","agent_name","agent_phone","country","county","description","details_url","displayable_address","first_published_date","floor_plan","image_caption","image_url","last_published_date","latitude","letting_fees","listing_id","listing_status","longitude","num_bathrooms","num_bedrooms","num_floors","num_recepts","outcode","post_town","price","price_change","property_type","rental_prices_accurate","rental_prices_per_month","rental_prices_per_week","short_description","status","street_name","thumbnail_url"])
			f.writerow( #[li["agent_address"], \
						#li["agent_logo"], \
						#li["agent_name"].encode(encoding='UTF-8'), \
						#li["agent_phone"], \
						[li["country"], \
						li["county"], \
						#li["description"].encode(encoding='UTF-8'), \
						#li["details_url"], \
						li["displayable_address"].encode(encoding='UTF-8'), \
						li["first_published_date"], \
						#li.get("floor_plan"), \
						#li["image_caption"], \
						#li["image_url"], \
						li["last_published_date"], \
						li["latitude"], \
						#li.get("letting_fees").encode(encoding='UTF-8'), \
						#li.get("letting_fees"), \
						li["listing_id"], \
						li["listing_status"], \
						li["longitude"], \
						li["num_bathrooms"], \
						li["num_bedrooms"], \
						li["num_floors"], \
						li["num_recepts"], \
						li["outcode"], \
						li["post_town"], \
						li["price"], \
						li.get("price_change"), \
						li["property_type"], \
						li["rental_prices"]["accurate"], \
						li["rental_prices"]["per_month"], \
						li["rental_prices"]["per_week"], \
						#li["short_description"].encode(encoding='UTF-8'), \
						li["status"], \
						li["street_name"].encode(encoding='UTF-8') ])
						#li["thumbnail_url"].encode(encoding='UTF-8')])

		param['page_number'] += 1

