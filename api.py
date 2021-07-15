from fastapi import FastAPI

"""
Soooo? We really doing this now?
okayyyy, Whatevvv
"""

app = FastAPI(title="API endpoints for virtual makeup",
              description="These API endpoints can be used to try virtual face makeup - lip-color, blush, foundation")


@app.get('/')
def root():
    return {"title": "Welcome",
            "message": "Nothing much to see here but HEY! try out the other endpoints. Hope you like them"}
