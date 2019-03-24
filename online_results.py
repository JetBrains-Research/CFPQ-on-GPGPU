from flask import Flask
import pandas as pd
app = Flask(__name__)

html_page="""
<head>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
</head>
<body>

<div class="container-fluid">
  <div class="row">
    <div class="col-8">{}</div>
    <div class="col-4">
      <textarea readonly style="width:100%; height:100%;">{}</textarea>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
</body>
"""

@app.route("/")
def get_results():
    data = pd.read_csv('result.csv')
    time_cols = [col for col in data.columns if 'all_mean' in col]
    with open('log.txt', 'r') as f:
        log = ''.join(f.readlines())
    return html_page.format(
        data[['Test name'] + time_cols].to_html(classes=['table-bordered', 'table-striped', 'table-hover', 'table-responsive'], index=False),
        log
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777)
