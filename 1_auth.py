"""
Stage 1 - File 1: Authenticate with Google Earth Engine.

Run this ONCE before anything else (and again only if your credentials expire).
It opens a browser / notebook flow to log in, then initializes the project.
"""

import ee

PROJECT_ID = "methaneplume"

ee.Authenticate(auth_mode="notebook")
ee.Initialize(project=PROJECT_ID)

print("Authenticated successfully. Project:", PROJECT_ID)
