/** Browser-side document parsing: PDF (pdf.js), DOCX (mammoth), TXT/MD. */

import { cleanText } from "./extract";

export const SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"];
export const MAX_BYTES = 10 * 1024 * 1024;

export class ParseError extends Error {}

export interface ParsedDocument {
  filename: string;
  text: string;
  pages: number;
  warnings: string[];
}

export function extensionOf(filename: string): string {
  const idx = filename.lastIndexOf(".");
  return idx === -1 ? "" : filename.slice(idx).toLowerCase();
}

export function validateFile(file: File): string | null {
  const ext = extensionOf(file.name);
  if (!SUPPORTED_EXTENSIONS.includes(ext)) return `Unsupported file type '${ext || "unknown"}'`;
  if (file.size === 0) return "File is empty";
  if (file.size > MAX_BYTES) return "File exceeds the 10 MB limit";
  return null;
}

async function parsePdf(buffer: ArrayBuffer): Promise<[string, number]> {
  const pdfjs = await import("pdfjs-dist");
  const workerSrc = (await import("pdfjs-dist/build/pdf.worker.min.mjs?url")).default;
  pdfjs.GlobalWorkerOptions.workerSrc = workerSrc;
  const doc = await pdfjs.getDocument({ data: new Uint8Array(buffer) }).promise;
  const parts: string[] = [];
  for (let i = 1; i <= doc.numPages; i += 1) {
    const page = await doc.getPage(i);
    const content = await page.getTextContent();
    parts.push(
      content.items
        .map((item) => ("str" in item ? item.str : ""))
        .join(" ")
        .replace(/\s{2,}/g, " "),
    );
  }
  return [parts.join("\n"), doc.numPages];
}

async function parseDocx(buffer: ArrayBuffer): Promise<[string, number]> {
  const mammoth = await import("mammoth/mammoth.browser");
  const result = await mammoth.extractRawText({ arrayBuffer: buffer });
  return [result.value, 1];
}

export async function parseFile(file: File): Promise<ParsedDocument> {
  const invalid = validateFile(file);
  if (invalid) throw new ParseError(invalid);

  const ext = extensionOf(file.name);
  let text = "";
  let pages = 1;
  try {
    if (ext === ".pdf") [text, pages] = await parsePdf(await file.arrayBuffer());
    else if (ext === ".docx") [text, pages] = await parseDocx(await file.arrayBuffer());
    else text = await file.text();
  } catch (error) {
    throw new ParseError(`Could not read document: ${(error as Error).message}`);
  }

  const cleaned = cleanText(text);
  if (cleaned.length < 50) {
    throw new ParseError("No extractable text found (document may be a scanned image)");
  }
  const warnings = cleaned.length < 300 ? ["Very little text extracted; extraction quality may be low."] : [];
  return { filename: file.name, text: cleaned, pages, warnings };
}
