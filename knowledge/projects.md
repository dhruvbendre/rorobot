---
category: Projects
title: Projects
source: knowledge/projects.md
date: 2026-09-05
project:
document_type: catalogue
status: live
---

# Projects

Dhruv Bendre's shipped projects, most of them built with Python and Streamlit and several of them live online.

## SnapClass (SmartSnapClass), AI-powered attendance system

SnapClass is an AI-powered full-stack attendance management platform that automates classroom attendance through facial recognition and voice authentication. It gives teachers a fast, secure and intuitive way to manage classes and attendance.

- Stack: Streamlit, an SVM classifier for facial recognition using the dlib library, and voice authentication powered by PyDub. Users are authenticated securely with password hashing.
- Teachers can create and manage classes; attendance is faster and more reliable than traditional methods.
- Design: clean, intuitive and user-friendly for both teachers and students. Every workflow was optimised to reduce manual effort, so users create classes, mark attendance and manage records with minimal clicks.
- Duration: three to four weeks. Year: 2026.
- Live demo: https://snapclassprojectpro.streamlit.app/

## Multi-Agent AI Research System

The Multi-Agent AI Research System is an intelligent research assistant that automates the entire research process using a team of specialised AI agents. It searches the web, analyses reliable sources, generates a structured report and reviews its own quality.

- Stack: LangChain, Groq's Llama 3.3 70B model, Tavily Search and BeautifulSoup.
- Architecture: a Search Agent discovers relevant sources with Tavily Search; a Reader Agent extracts clean content from selected web pages with BeautifulSoup; a Writer Agent synthesises the information into a professional research report; a Critic Agent evaluates the report by assigning a score, identifying strengths and weaknesses, and suggesting improvements.
- Why it matters: separating responsibilities across dedicated agents delivers more accurate, organised and trustworthy research than a single LLM response, and the modular design can be extended with fact-checking, citation validation or summarisation agents.
- Duration: three to four weeks. Year: 2026. Client: self-initiated.
- Live demo: https://multiagent-research-systembydhruvb.streamlit.app/

## Hackathon Management Platform (Get Set Learn)

The Hackathon Management Platform was built as a Get Set Learn assignment to simplify the entire hackathon experience for students, teachers and organisers.

- Stack: Python, Streamlit, Supabase, RAG and Agno.
- Features: participants discover hackathons, register in a few clicks, complete payments securely and instantly download payment receipts inside the application.
- Get Set Learn AI Assistant: a Retrieval-Augmented Generation (RAG) chatbot trained on hackathon documentation, FAQs and guidelines. Users ask natural-language questions about registration, eligibility, schedules, rules or judging criteria and get accurate, context-aware answers in seconds.
- Design: playful and engaging, inspired by educational products for K–12 students, with vibrant colours, bold typography, custom illustrations and interactive layouts. Every screen reduces complexity while keeping a modern look, so first-time participants can explore confidently.
- Duration: one week. Year: 2026. Client: Get Set Learn.
- Live demo: https://stem-gsl-hack.streamlit.app/

## MyChessBot

My Chess Bot is an AI-powered chess application that learns Dhruv's playing style from his personal PGN game database and combines that model with the Stockfish engine to deliver accurate, human-like gameplay through an interactive Streamlit interface. It keeps strong, accurate play while preserving his unique strategies and move preferences.

- Stack: Python, a machine learning model trained on PGN games, Stockfish, Streamlit.
- Status: started in 2024 and ongoing. The interface is being redesigned for a clean, modern and immersive experience with real-time game analysis; a fully functional live demo will be available once the design is complete.
- Source: https://github.com/bscitdhruvbendre-create/chessBot
