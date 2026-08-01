import markdown
import os

md_path = "/home/zera/.gemini/antigravity/brain/713ce5f7-0a7c-4b14-b143-da5031b8d375/artifacts/Q1_Manuscript_Results_Draft.md"
out_path = "/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer/Q1_Manuscript_Results.doc"

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

# Generate HTML with tables extension
html = markdown.markdown(text, extensions=['tables'])

styled_html = f"""
<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: 'Times New Roman', serif; font-size: 12pt; line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; margin-bottom: 20px; }}
th, td {{ border: 1px solid black; padding: 8px; text-align: center; vertical-align: middle; }}
th {{ background-color: #f2f2f2; font-weight: bold; }}
h1 {{ font-size: 18pt; font-weight: bold; margin-top: 24px; }}
h2 {{ font-size: 14pt; font-weight: bold; margin-top: 20px; }}
h3 {{ font-size: 12pt; font-weight: bold; margin-top: 16px; }}
</style>
</head>
<body>
{html}
</body>
</html>
"""

with open(out_path, "w", encoding="utf-8") as f:
    f.write(styled_html)

print(f"Created Word document at {out_path}")
