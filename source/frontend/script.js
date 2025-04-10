const form = document.getElementById("predictionForm");
const resultBox = document.getElementById("result");
const battingTeamInput = document.getElementById("battingTeam");
const bowlingTeamInput = document.getElementById("bowlingTeam");
const venueInput = document.getElementById("venue");
const oversLeftInput = document.getElementById("oversLeft");
const wicketsLeftInput = document.getElementById("wicketsLeft");
const currentScoreInput = document.getElementById("currentScore");
const spinner = document.getElementById("spinner");
console.log("Submit clicked");


// Utility functions
function showError(input, message) {
  const error = input.parentElement.querySelector(".error-message");
  error.textContent = message;
  error.style.display = "block";
}

function clearError(input) {
  const error = input.parentElement.querySelector(".error-message");
  error.textContent = "";
  error.style.display = "none";
}

function validateOversLeft(value) {
  const pattern = /^([0-9]|1[0-9])(\.[0-5])?$/;
  return pattern.test(value);
}

function validateInteger(value, min, max) {
  const intVal = parseInt(value);
  return Number.isInteger(intVal) && intVal >= min && intVal <= max;
}

// Real-time validation
battingTeamInput.addEventListener("change", () => {
  if (battingTeamInput.value === bowlingTeamInput.value) {
    showError(bowlingTeamInput, "Bowling team must be different from batting team.");
    showError(battingTeamInput, "Batting team must be different from bowling team.");
  } else {
    clearError(bowlingTeamInput);
    clearError(battingTeamInput);
  }
});

bowlingTeamInput.addEventListener("change", () => {
  if (bowlingTeamInput.value === battingTeamInput.value) {
    showError(bowlingTeamInput, "Bowling team must be different from batting team.");
    showError(battingTeamInput, "Batting team must be different from bowling team.");
  } else {
    clearError(bowlingTeamInput);
    clearError(battingTeamInput);
  }
});

oversLeftInput.addEventListener("input", () => {
  const value = oversLeftInput.value.trim();
  if (!validateOversLeft(value)) {
    showError(oversLeftInput, "Enter overs like 4.3 (max 19.5)");
  } else {
    clearError(oversLeftInput);
  }
});

wicketsLeftInput.addEventListener("input", () => {
  const value = wicketsLeftInput.value.trim();
  if (!validateInteger(value, 0, 10)) {
    showError(wicketsLeftInput, "Wickets must be an integer between 0 and 10.");
  } else {
    clearError(wicketsLeftInput);
  }
});

currentScoreInput.addEventListener("input", () => {
  const value = currentScoreInput.value.trim();
  if (!validateInteger(value, 0, 200)) {
    showError(currentScoreInput, "Score must be an integer between 0 and 200.");
  } else {
    clearError(currentScoreInput);
  }
});

// Submit logic
form.addEventListener("submit", (e) => {
  e.preventDefault();

  const battingTeam = battingTeamInput.value;
  const bowlingTeam = bowlingTeamInput.value;
  const venue = venueInput.value.trim();
  const oversLeft = oversLeftInput.value.trim();
  const wicketsLeft = wicketsLeftInput.value.trim();
  const currentScore = currentScoreInput.value.trim();

  let hasError = false;

  // Validate before proceeding
  if (battingTeam === bowlingTeam) {
    showError(bowlingTeamInput, "Bowling team must be different from batting team.");
    showError(battingTeamInput, "Batting team must be different from bowling team.");
    hasError = true;
  }

  if (!validateOversLeft(oversLeft)) {
    showError(oversLeftInput, "Enter overs like 4.3 (max 19.5)");
    hasError = true;
  }

  if (!validateInteger(wicketsLeft, 0, 10)) {
    showError(wicketsLeftInput, "Wickets must be an integer between 0 and 10.");
    hasError = true;
  }

  if (!validateInteger(currentScore, 0, 200)) {
    showError(currentScoreInput, "Score must be an integer between 0 and 200.");
    hasError = true;
  }

  if (hasError) return;

  // Show spinner
  spinner.style.display = "inline-block";
  resultBox.innerHTML = "";

  // Simulate delay (e.g. API call)
  setTimeout(() => {
    spinner.style.display = "none";
    const oversWhole = Math.floor(parseFloat(oversLeft));
    const oversBalls = Math.round((parseFloat(oversLeft) - oversWhole) * 10);
    const totalBalls = oversWhole * 6 + oversBalls;

    const predictedScore = parseInt(currentScore) + Math.floor(totalBalls * 1.33 + parseInt(wicketsLeft));
    resultBox.innerHTML = `<h3>Predicted Final Score:</h3><p>${predictedScore} runs</p>`;
  }, 1500);
});
