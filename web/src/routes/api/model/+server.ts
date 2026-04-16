import { json } from "@sveltejs/kit";

import { resolveModelPayload } from "$lib/server/huggingface";

export type ErrorResponse = {
  message: string;
};

export type ModelResponse = Awaited<ReturnType<typeof resolveModelPayload>>;

export type ApiResponse =
  | {
      success: true;
      data: ModelResponse;
    }
  | {
      success: false;
      error: ErrorResponse;
    };

export async function POST({ request }) {
  let body: any;

  try {
    body = await request.json();
  } catch {
    return json(
      {
        success: false,
        error: { message: "Send JSON with a repo field." },
      } satisfies ApiResponse,
      { status: 400 },
    );
  }

  try {
    const payload: ModelResponse = await resolveModelPayload(body.repo);
    return json({ success: true, data: payload } satisfies ApiResponse);
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Unable to resolve model metadata.";
    return json(
      { success: false, error: { message: message } } satisfies ApiResponse,
      { status: 400 },
    );
  }
}
