document.addEventListener('DOMContentLoaded', function () {
    const filePath = localStorage.getItem('filePath'); // Store the file path after uploading/downloading
    if (!filePath) {
        alert('No file path found. Please upload or download a file first.');
        return;
    }

    fetch('/get_indicators', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `file_path=${filePath}`
    })
    .then(response => response.json())
    .then(data => {
        const indicators = Object.keys(data);
        const indicatorCharts = document.getElementById('indicator-charts');

        indicators.forEach(indicator => {
            const chartContainer = document.createElement('div');
            chartContainer.classList.add('col-md-6', 'mb-4');
            chartContainer.innerHTML = `
                <h4>${indicator}</h4>
                <canvas id="${indicator}-chart" width="400" height="200"></canvas>
            `;
            indicatorCharts.appendChild(chartContainer);

            const ctx = document.getElementById(`${indicator}-chart`).getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({ length: data[indicator].length }, (_, i) => i + 1),
                    datasets: [
                        {
                            label: indicator,
                            data: data[indicator],
                            borderColor: 'blue',
                            borderWidth: 2,
                            fill: false,
                            tension: 0.4,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time'
                            },
                            grid: {
                                display: true
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Value'
                            },
                            grid: {
                                display: true
                            }
                        }
                    }
                }
            });
        });
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while fetching indicators.');
    });
});
