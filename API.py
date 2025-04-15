import time

from flask import Flask, jsonify
from Analyzer import GamblingAnalyzer

analyzer = GamblingAnalyzer()
app = Flask(__name__)


@app.route('/check_url/<path:url>', methods=['GET'])
def check_url(url):
    start = time.time()
    block = analyzer.predict(url)
    elapsed = round(time.time() - start, 3)

    print(f"[Checked] {url} → Block: {block} in {elapsed}s")
    return jsonify({
        'url': url,
        'gambling': block,
        'elapsed': elapsed
    })



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
