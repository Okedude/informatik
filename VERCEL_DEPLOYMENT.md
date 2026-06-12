Steps to deploy to Vercel

1) Push this repository to GitHub (if not already).
2) Create a Vercel account and import the GitHub repository: https://vercel.com/new
3) In the Vercel Project Settings -> Environment Variables, set:
   - GEMINI_API_KEY (optional)
   - GEMINI_MODEL (optional)
   - HUGGINGFACE_API_KEY (recommended if no Gemini access)
   - HUGGINGFACE_MODEL (optional, default: mistralai/mistral-small)
   - PORT is not needed on Vercel
4) Deploy. The API is available at `https://<your-project>.vercel.app/api/chat` and the website at the project root.

Security: store keys in Vercel project Environment Variables (do NOT commit them to the repo).

If you want, I can create a GitHub Action or help connect the repo to Vercel.