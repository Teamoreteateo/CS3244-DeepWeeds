## CS3244-DeepWeeds

# Run these comments on your local machine:
cd ~/Documents
git clone https://github.com/<A1-username>/CS3244-DeepWeeds.git
cd CS3244-DeepWeeds
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data outputs figures


# Best Practices
git pull
git checkout -b alex/feature-extraction     # create and switch to a branch

Work, then:

git add src/features.py
git commit -m "xyz"
git push -u origin <branch>

The -u origin <branch> is needed the first time you push a new branch; afterwards plain git push works.

[Approve new commits on Github]

Then get back to main:

git checkout main
git pull
git branch -d alex/feature-extraction    # deletes branch