                     
"""
OAuth 2.0 Authentication Setup Script for AarogyaVeda
Run this once to authorize Google Drive access
"""

from google_drive_manager import GoogleDriveManager

print("=" * 60)
print("AarogyaVeda - Google Drive OAuth 2.0 Setup")
print("=" * 60)
print()
print("Initializing OAuth 2.0 authentication...")
print("A browser window will open shortly for authorization.")
print()

try:
    gm = GoogleDriveManager()
    print()
    print("✅ Authentication successful!")
    print()
    
                                      
    files = gm.get_file_list(limit=5)
    print(f"✅ Google Drive connected! Found {len(files)} files in your archive.")
    
    if files:
        print()
        print("Recent files:")
        for file in files[:3]:
            print(f"  • {file['name']}")
    
    print()
    print("=" * 60)
    print("✅ Setup complete! You can now use AarogyaVeda.")
    print("=" * 60)
    print()
    print("token.json has been created and saved securely.")
    print("Next time you run the app, no browser authorization needed!")
    
except Exception as e:
    print()
    print(f"❌ Error during authentication: {e}")
    print()
    print("Troubleshooting:")
    print("1. Make sure google_service_account.json exists in your project folder")
    print("2. Verify it's the OAuth 2.0 Desktop Client credentials JSON")
    print("3. Check your internet connection")
    print()
    raise
