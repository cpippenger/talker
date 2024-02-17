from flask import Flask, jsonify, send_file
# ... (Other imports)

# ... (Your existing code)

# Create Swagger documentation
@app.route('/swagger.json')
def swagger():
    return send_file('swagger.json')  # Assuming you generate swagger.json in the same directory

# ... (Your existing code)

if __name__ == '__main__':
    app.run(debug=True)