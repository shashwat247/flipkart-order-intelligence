/** Renders the agent's answer, honouring the **bold** spans its templates emit.
 *
 * Deliberately not a markdown library: the only markup Part 3 produces is bold,
 * and building React nodes from a split keeps the answer text as text — there
 * is no HTML injection path for anything the agent or a retrieved document says.
 */
export function AnswerText({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <p className="whitespace-pre-wrap font-body text-sm leading-relaxed text-paper">
      {parts.map((part, i) =>
        part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
          <strong key={i} className="font-semibold text-white">
            {part.slice(2, -2)}
          </strong>
        ) : (
          part
        ),
      )}
    </p>
  );
}
