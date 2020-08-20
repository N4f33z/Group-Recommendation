from ibm_watson import PersonalityInsightsV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from os.path import join, dirname
import json

#Initialized for API usage
authenticator = IAMAuthenticator('rKKOOaC5nd8haGRmAAFFeCXt196bYI5qhWogSapG_zlJ')
personality_insights = PersonalityInsightsV3(
    version='2017-10-13',  #2020-01-30
    authenticator=authenticator
)

personality_insights.set_service_url('https://api.us-south.personality-insights.watson.cloud.ibm.com/instances/66b10b97-021d-47ae-b5c2-fec4f691fc69')



#Call for value scores through IBM API
with open(join(dirname(__file__), './051Craig.csv')) as profile_csv:
    profile = personality_insights.profile(
        profile_csv.read(),
        'application/json',
        content_type='text/plain',
        consumption_preferences=True,
        raw_scores=True
    ).get_result()
print(json.dumps(profile, indent=2))

print(profile["values"][0]["raw_score"])
print(profile["values"][1]["raw_score"])
print(profile["values"][2]["raw_score"])
print(profile["values"][3]["raw_score"])
print(profile["values"][4]["raw_score"])

