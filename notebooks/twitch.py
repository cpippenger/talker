from twitchAPI.twitch import Twitch
from twitchAPI.helper import first
import asyncio

client_id = "wwe5w9lnob78v1ph2xhefpyk6zi2k3"
secret = "1vk4sy6jx1lv6s5b673gvdqt9nlu3v"

async def twitch_example():
    # initialize the twitch instance, this will by default also create a app authentication for you
    twitch = await Twitch(client_id, secret)
    # call the API for the data of your twitch user
    # this returns a async generator that can be used to iterate over all results
    # but we are just interested in the first result
    # using the first helper makes this easy.
    user = await first(twitch.get_users(logins='your_twitch_user'))
    # print the ID of your user or do whatever else you want with it
    print(user.id)
    await twitch.close()

# run this example
asyncio.run(twitch_example())