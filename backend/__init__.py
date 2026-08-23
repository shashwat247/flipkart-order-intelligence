"""HTTP backend for the Flipkart support agent.

A transport layer only. Every endpoint delegates to the real Part 1/2/3 code —
the saved Random Forest, the saved ResNet-18 and the LangGraph agent — and
returns what they produce. Nothing here recomputes, caches or fakes a model
output, and there is no canned question/answer table anywhere in this package.
"""
