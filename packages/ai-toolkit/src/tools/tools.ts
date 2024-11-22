import {tool} from "@langchain/core/tools";
import {z} from "zod";

import * as process from "node:process";
import {placesSearchTool} from "../graphs/rag/places";

interface JourneyResponse {
  journeys: Array<{
    duration: number;
    sections: Array<any>;
    arrival_date_time: string;
    departure_date_time: string;
  }>;
}

async function callIDFMAPI(fields: {
  fromLong: string;
  fromLat: string;
  toLong: string;
  toLat: string;
  datetime: string;
}): Promise<JourneyResponse> {
  const baseURL = "https://prim.iledefrance-mobilites.fr/marketplace/v2/navitia/journeys";
  const destination = `${fields.fromLong}%3B%20${fields.fromLat}&to=${fields.toLong}%3B%20${fields.toLat}&datetime=${fields.datetime}`;
  const url = `${baseURL}?from=${destination}`;

  const response = await fetch(url, {
    method: "GET",
    headers: {
      'Accept': 'application/json',
      'apikey': process.env.PRIM_API_KEY
    } as HeadersInit
  });

  if (!response.ok) {
    let res: string;
    try {
      res = JSON.stringify(await response.json(), null, 2);
    } catch (_) {
      res = await response.text();
    }
    throw new Error(`Failed to fetch journey data.\nResponse: ${res}`);
  }

  return await response.json();
}

export const journeyPlannerTool = tool(
  async (input) => {
    try {
      const data = await callIDFMAPI({
        fromLong: input.fromLong,
        fromLat: input.fromLat,
        toLong: input.toLong,
        toLat: input.toLat,
        datetime: input.datetime
      });
      console.log(data);
      return JSON.stringify(data, null, 2);
    } catch (e: any) {
      console.warn("Error fetching journey data", e.message);
      return `An error occurred while fetching journey data: ${e.message}`;
    }
  },
  {
    name: "journey_planner",
    description: "Plans journeys using the Île-de-France Mobilités API. Returns available routes between two points with timing information. Use the places_search tool to find the gps coordinates",
    schema: z.object({
      fromLong: z.string().describe("Departure longitude"),
      fromLat: z.string().describe("Departure latitude"),
      toLong: z.string().describe("Arrival longitude"),
      toLat: z.string().describe("Arrival latitude"),
      datetime: z.string().describe("Journey datetime in format YYYYMMDDTHHMMSS")
    })
  }
);

// Add to tools list
export const ALL_TOOLS_LIST = [
  journeyPlannerTool,
  placesSearchTool
];
