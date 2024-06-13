// frontend/script.js
async function fetchForecast(indicator) {
  const response = await fetch(`http://127.0.0.1:5000/forecast/${indicator}`);
  const data = await response.json();
  displayForecast(data, indicator);
}

function displayForecast(data, indicator) {
  const resultDiv = document.getElementById("forecast-result");
  resultDiv.innerHTML = `
        <h2>${indicator.toUpperCase()} Forecast</h2>
        <pre>${JSON.stringify(data, null, 2)}</pre>
    `;
}
