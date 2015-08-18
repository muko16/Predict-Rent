import requests
import json

host = "http://api.zoopla.co.uk/api/v1/"
action = "property_listings.js"
param= {}
param['api_key'] = "hkp7jznhuq4skfb9chsn8sw3"
#param['area'] = "London"
param['postcode'] = "N1"
#param['page_size'] = 100  		# max:100 default:10
#param['page_number'] = 1  		# default:1
param['listing_status'] = "rent" # "sale" or "rent"
#param['include_sold'] = 1 		# Defaults to 0
#param['include_rented'] = 1 		# Defaults to 0

#http://api.zoopla.co.uk/api/v1/property_listings.xml?area=London&api_key=hkp7jznhuq4skfb9chsn8sw3
#print param

wshost = "http://api.walkscore.com/score?"
wsapikey = "43a2ee9b8b5fb11b7e54865e15dca18a" # walkscore api key

endpoint = host + action + "?" + '&'.join("%s=%s" % (key,val) for (key,val) in param.iteritems())
print endpoint

r = requests.get(endpoint)
data = json.loads(r.content)
#print json.dumps(data, indent=4, sort_keys=True)

for listing in data["listing"]:
	print listing
	print ""
	print "[Adrress] " + listing["displayable_address"]
	#print "  # of bedrooms = " + listing["num_bedrooms"] + ", price = " + listing["price"] + ", listing_status = " + listing["listing_status"]
	print "  # of bedrooms: " + listing["num_bedrooms"]
	print "  price: " + listing["price"]
	print "  listing_status: " + listing["listing_status"]
	
	wsparam = {}
	wsparam['wsapikey'] = wsapikey
	wsparam['address'] = listing["displayable_address"]
	wsparam['lat'] = listing["latitude"]
	wsparam['lon'] = listing["longitude"]
	wsparam['format'] = "json"

	endpoint = wshost + '&'.join("%s=%s" % (key,val) for (key,val) in wsparam.iteritems())
	r = requests.get(endpoint)
	data = json.loads(r.content)
	print "  walkscore: " + str(data["walkscore"]) + " (" + data["description"] + ")"
	#print json.dumps(data, indent=4, sort_keys=True)
	print ""




