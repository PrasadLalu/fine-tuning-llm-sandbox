# Fine Tuning LLM Sandbox

## Create Python Virtual Environment
- **Install `uv`:**

   ```bash
   pip install uv

   uv --version
- **Create a Virtual Environmen:**
    ```bash
    uv venv .venv --python=3.10
- **Active Virtual Environment:**
    ```bash
    .venv\Scripts\activate
- **Initialize the `uv` Project:**
    ```bash
    uv init
- **Add Dependencies (e.g. add `python-dotenv`):**
    ```bash
    uv add python-dotenv

## Run:
- Streamlit app:
```
uv run streamlit run main.py --server.runOnSave true
```
