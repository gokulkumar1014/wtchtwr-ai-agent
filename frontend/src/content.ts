export const WELCOME_MESSAGE = `Hi there! I'm wtchtwr 👋

You can ask me things like:
• "Average occupancy rate for next 60 days for my listings in Manhattan"
• "Which of my properties have paid parking?"
• "Compare avg price in Midtown: mine vs market"
• "Show the latest 5 reviews for listing 2595"

Useful shortcuts: Type help for knowing what things you can do using the chatbot · dashboard for viewing the dashboard · slackbot for going to the slack app · export for jumping into the data export page. How can I help today?`;

export const HELP_INTRO = `wtchtwr blends NL→SQL, review retrieval, and sentiment analysis so you can pivot between metrics, narratives, and guest tone without leaving chat. Dial in boroughs, property types, hosts, time horizons, and more to tailor every follow-up.`;

export const SAMPLE_QUERIES: string[] = [
  `"Average occupancy rate for next 60 days for my listings in Manhattan"`,
  `"Compare avg price in Midtown: mine vs market"`,
  `"Give me a concise summary of the last five conversations with the bot"`,
  `"Which of my properties have paid parking?"`,
  `"Show the latest 5 reviews for listing 2595"`,
  `"List my listings with occupancy below 40% last month"`,
  `"Which Highbury listing delivered the highest revenue in Queens last quarter?"`,
  `"Highlight the amenities guests mention most often in reviews for listing 2595"`,
  `"What percentage of Highbury listings allow pets?"`,
  `"Benchmark Williamsburg vs Chelsea occupancy for the next 60 days"`,
  `"Summarise guest sentiment about cleanliness for listing 2595"`,
  `"Pull recent review highlights mentioning 'noise' near Times Square"`,
  `"Compare average booking lead time: Highbury vs market"`,
  `"Which of my listings have occupancy above 85% this month?"`,
  `"Average revenue per available room for SoHo lofts"`,
  `"What is the cancellation rate for competitors in Brooklyn over the last 60 days?"`,
  `"Which Highbury listings include a dedicated workspace?"`,
  `"Show weekend vs weekday pricing for listing 2595 last month"`,
  `"Identify listings with more than three negative reviews about Wi-Fi"`,
  `"Share an amenities comparison between my listings and the market in Brooklyn"`,
  `"Surface the most positive guest quotes about our Williamsburg lofts"`,
  `"Analyse review sentiment for parking across Highbury properties"`,
  `"Which hosts in Harlem have better occupancy than Highbury in the next 90 days?"`,
  `"Compare occupancy and revenue for Highbury vs market across the next 30, 60, 90, and 365 days"`,
  `"Give me the top five sentiment trends across all reviews this quarter"`,
  `"What are the most common amenity gaps compared to the market in Chelsea?"`,
  `"Summarise RAG findings about guest experience in Upper West Side"`,
  `"Show the 10 newest bookings for listing 3021"`,
  `"Compare ADR for entire-home listings in Brooklyn across Highbury and the market"`,
  `"List competitors with better review scores but lower prices in Tribeca"`,
];

export const ABOUT_POINTS: { title: string; detail: string }[] = [
  {
    title: "What wtchtwr Solves",
    detail:
      "wtchtwr is a portfolio intelligence co-pilot for STR operators. It combines NL→SQL analytics, RAG-powered review retrieval, sentiment scoring, portfolio triage, competitor benchmarking, and amenity diagnostics so you can interrogate every KPI from one prompt bar.",
  },
  {
    title: "End-to-End Operator Workflow",
    detail:
      "From the chat canvas you can run SQL-grade pricing & occupancy checks, pull review citations, summarize chats, open recent messages, email stakeholders, export CSVs, and trigger Slackbot, dashboard, or data export flows without leaving wtchtwr.",
  },
  {
    title: "Signals + Narrative",
    detail:
      "Every answer blends structured DuckDB metrics with Qdrant review snippets so you get verifiable citations for SQL queries and qualitative insights (parking, comfort, amenities, sentiment themes). Portfolio Triage and competitor benchmarking surface risks and pricing gaps automatically.",
  },
  {
    title: "Channels & Surfaces",
    detail:
      "The React + Vite dashboard, Slackbot, chat history, recent-message spotlight, and export workspace are all wired into the same LangGraph pipeline. Start in chat, continue in Slack, or pull CSV/email packages as soon as a question lands.",
  },
];
