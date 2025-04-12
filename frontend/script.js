async function classify() {
    const comment = document.getElementById("comment").value;
    const response = await fetch("http://localhost:8000/classify/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comment }),
    });
    const data = await response.json();
    document.getElementById("result").innerHTML = `<strong>Detected Interests:</strong> ${data.interests.join(', ')}`;
  }
  