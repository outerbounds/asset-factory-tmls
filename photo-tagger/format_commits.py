import subprocess
import json
from datetime import datetime

def get_git_log_json():
    # Format: SHA | Timestamp | Author | Subject | Body
    format_str = "%h|%aI|%ae|%s|%b"
    result = subprocess.run(
        ["git", "log", f"--pretty=format:{format_str}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    commits = []
    for line in result.stdout.split("\n"):
        parts = line.split("|", 4)
        if len(parts) < 5:
            continue
        sha, timestamp, email, title, body = parts

        commit = {
            "commit_sha": sha,
            "container_image": f"some.container:image:{sha}",  # Placeholder logic
            "owner": email,
            "pr_description": body.strip().replace("\n", " "),
            "pr_title": title.strip(),
            "timestamp": timestamp
        }
        commits.append(commit)

    return commits

import json
print(json.dumps(get_git_log_json(), indent=2))
