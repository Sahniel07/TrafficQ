import requests
import json

url = "https://disrupt.sdk.efs.ai/api/graphql"
basic_auth = "Authorization: Basic aGFja2F0aG9uQHRoaS5kZTpoYWNrYXRob24yMDI1"

body = """
query entities {
  entities(source: 24,classes : ["car"], after: "2023-09-24T00:00:00.016482+00:00" , first:600) {
    edges {
      node {
        id
        class {
            color
            name
            description
        }
        trajectory {
            time
            coordinateLongLat
        }
        averageVelocity
      }
      
    }
  }
}
"""

# Request
response = requests.post(
    url=url, 
    json={"query": body}, 
    headers={"Authorization": basic_auth})
print("response status code: ", response.status_code)

# Print response
if response.status_code == 200:
    # save the response to a file
    with open('data_2.json', 'w') as f:
        json.dump(response.json(), f)
    
    # data = json.loads(response.content)
    # print(json.dumps(data, indent = 4))