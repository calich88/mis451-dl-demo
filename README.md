# Deploy a Streamlit App to Streamlit Community Cloud

A beginner-friendly guide to publishing your Python app **for free** with [Streamlit Community Cloud](https://share.streamlit.io/). The platform connects directly to GitHub, so **no command-line Git experience is required**.

> **Before you begin:** Have your `app.py` file ready and create a free [GitHub](https://github.com/) account if you do not already have one.

## Step 1 — Create `requirements.txt`

Streamlit Community Cloud runs your app on **Linux**. Create a file named `requirements.txt` in the same folder as `app.py` and add these **seven dependencies**:

```txt
streamlit
numpy
pandas
scikit-learn
tensorflow-cpu
matplotlib
openpyxl
```

**Important notes**

- **Do not blindly copy the output of `pip freeze` from a Windows environment.** It may include Windows-specific packages, such as `pywin32` or `tensorflow-intel`, that are unsuitable for a Linux deployment.
- Use **`tensorflow-cpu`** for this demo, which does not require GPU acceleration and avoids installing the GPU-oriented TensorFlow dependencies.
- These are unpinned package names for a simple introductory demo. For reproducible deployments, test your app and pin compatible package versions.

Your project should look like this:

```text
mis451-dl-demo/
├── app.py
├── requirements.txt
└── README.md              # Optional: this guide
```

## Step 2 — Upload Your Project to GitHub

You can complete this step entirely in your browser.

1. Go to [github.com](https://github.com/) and sign in.
2. Click the **+** icon in the top-right corner and select **New repository**.
3. Enter a repository name, for example, `mis451-dl-demo`.
4. Select **Public** and click **Create repository**.
5. Select **uploading an existing file** (or use **Add file → Upload files** if the repository already has files).
6. Drag and drop `app.py` and `requirements.txt`. You may also upload this `README.md`.
7. Click **Commit changes** to save your files.

> **Check:** Confirm that `app.py` and `requirements.txt` are visible at the root of your GitHub repository.

## Step 3 — Deploy on Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io/) and sign in using your GitHub account.
2. Click **Create app**.
3. When prompted, select **Yup, I have an app**.
4. Enter your deployment settings:

   | Setting | Example |
   | --- | --- |
   | **Repository** | `your-username/mis451-dl-demo` |
   | **Branch** | `main` |
   | **Main file path** | `app.py` |
   | **App URL** (optional) | `mis451-dl-demo` |

5. Open **Advanced settings** and choose **Python 3.11 or 3.12** for this demo. Click **Save**.
6. Click **Deploy**.
7. Monitor the deployment logs. Installation typically takes a few minutes; it may take longer when installing TensorFlow.

### Why select a Python version?

Your packages must support the Python version used on the server. For this tutorial, **Python 3.11 or 3.12** is a conservative, compatible choice. Newer TensorFlow releases may support newer versions of Python, but compatibility depends on the specific release. **Choose the Python version through Streamlit's Advanced settings** rather than assuming a `runtime.txt` file will configure it.

> **Important:** Streamlit Community Cloud does not let you change an existing app's Python version in place. To change it later, you must delete and redeploy the app.

## Step 4 — Share Your App

Once deployment succeeds, Streamlit gives you a public link similar to:

**https://mis451-dl-demo.streamlit.app**

This is an **illustrative URL**. Your actual app URL depends on the subdomain you select or the address assigned by Streamlit.

Share the link with classmates, your lecturer, or other users. When you commit updates to your GitHub code, Streamlit Community Cloud normally updates the deployed app automatically.

## Troubleshooting

| Issue | What to do |
| --- | --- |
| `ModuleNotFoundError` | Make sure every imported third-party library is included in `requirements.txt`. |
| `No matching distribution found` | Check package names and Python compatibility; remove Windows-only packages. |
| TensorFlow fails to install | Confirm that you selected Python 3.11 or 3.12 and used `tensorflow-cpu`. Review the deployment logs. |
| App crashes or runs slowly | Inspect the logs and reduce memory-intensive model loading or training. Free hosting has resource limits. |
| Updates do not appear | Verify that changes were committed to the deployed GitHub branch and check the Streamlit logs. |

### Free Hosting: Sleep Mode

Streamlit Community Cloud automatically puts apps to sleep after **12 hours without traffic**. This is normal and does not delete your project. Anyone with access can visit the app and click **“Yes, get this app back up!”** to wake it.

## Official Documentation

- [Streamlit Community Cloud — Overview](https://docs.streamlit.io/deploy/streamlit-community-cloud)
- [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Manage app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Manage your app and hibernation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [TensorFlow installation and Python compatibility](https://www.tensorflow.org/install/pip)

---

**Recommended for:** MIS 451 — Machine Learning for Business Analytics | Streamlit Deep Learning Demo
