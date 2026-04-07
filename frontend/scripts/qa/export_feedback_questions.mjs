#!/usr/bin/env node

/**
 * Extract unique questions from feedback JSON export
 * 
 * Usage:
 *   node export_feedback_questions.mjs
 *   node export_feedback_questions.mjs --input path/to/feedback.json --output path/to/output.json --min-length 5
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// CLI argument parsing
const args = process.argv.slice(2);
let inputPath = 'data/20260402_feedback_export.json';
let outputPath = '.sisyphus/evidence/ui-qa-9098/questions.json';
let minLength = 3;

function showHelp() {
  console.log(`Extract unique questions from feedback JSON export

Usage: node export_feedback_questions.mjs [options]

Options:
  --input <path>     Input JSON file (default: data/20260402_feedback_export.json)
  --output <path>    Output JSON file (default: .sisyphus/evidence/ui-qa-9098/questions.json)
  --min-length <n>  Minimum question length (default: 3)
  --help            Show this help message

Examples:
  node export_feedback_questions.mjs
  node export_feedback_questions.mjs --input custom/feedback.json --min-length 5`);
  process.exit(0);
}

// Parse arguments
for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if (arg === '--help' || arg === '-h') {
    showHelp();
  } else if (arg === '--input' && i + 1 < args.length) {
    inputPath = args[++i];
  } else if (arg === '--output' && i + 1 < args.length) {
    outputPath = args[++i];
  } else if (arg === '--min-length' && i + 1 < args.length) {
    const val = parseInt(args[++i], 10);
    if (isNaN(val) || val < 0) {
      console.error('Error: --min-length must be a non-negative integer');
      process.exit(1);
    }
    minLength = val;
  } else {
    console.error(`Error: Unknown option "${arg}". Use --help for usage.`);
    process.exit(1);
  }
}

// Resolve paths relative to repo root
const repoRoot = resolve(__dirname, '..', '..', '..');
const resolvedInput = resolve(repoRoot, inputPath);
const resolvedOutput = resolve(repoRoot, outputPath);

// Load and parse input JSON
let feedbackData;
try {
  const inputContent = readFileSync(resolvedInput, { encoding: 'utf-8' });
  feedbackData = JSON.parse(inputContent);
} catch (err) {
  if (err.code === 'ENOENT') {
    console.error(`Error: Input file not found: ${resolvedInput}`);
    console.error('Hint: Use --input to specify a different file, or run:');
    console.error(`  curl -fsS "http://localhost:8011/api/feedback/export/json?min_score=0" -o ${inputPath}`);
    process.exit(1);
  }
  if (err instanceof SyntaxError) {
    console.error(`Error: Invalid JSON in ${resolvedInput}: ${err.message}`);
    process.exit(1);
  }
  throw err;
}

// Validate input is an array
if (!Array.isArray(feedbackData)) {
  console.error(`Error: Input file must contain a JSON array, got: ${typeof feedbackData}`);
  process.exit(1);
}

// Extract and filter questions
const seen = new Set();
const questions = [];

for (const item of feedbackData) {
  // Rule 1: exclude non-string user_text
  if (typeof item.user_text !== 'string') {
    continue;
  }
  
  const trimmed = item.user_text.trim();
  
  // Rule 2: trim; exclude empty
  if (trimmed.length === 0) {
    continue;
  }
  
  // Rule 3: exclude length < min-length
  if (trimmed.length < minLength) {
    continue;
  }
  
  // Rule 4: dedupe by exact string, keeping first-seen order
  if (!seen.has(trimmed)) {
    seen.add(trimmed);
    questions.push(trimmed);
  }
}

// Ensure output directory exists
const outputDir = dirname(resolvedOutput);
mkdirSync(outputDir, { recursive: true });

// Write output JSON (pretty-printed, indent=2, UTF-8)
writeFileSync(resolvedOutput, JSON.stringify(questions, null, 2), { encoding: 'utf-8' });

console.log(`Extracted ${questions.length} unique questions from ${feedbackData.length} feedback entries`);
console.log(`Output saved to: ${resolvedOutput}`);