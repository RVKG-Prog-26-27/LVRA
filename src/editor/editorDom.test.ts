// @vitest-environment jsdom
import { extractText, renderSegments, unwrapEditedMarks } from './editorDom';

describe('result editor', () => {
  const setup = () => {
    const editor = document.createElement('div');
    renderSegments(editor, [
      { text: 'Es redzu ', uncertain: false },
      { text: 'kaķi', uncertain: true },
      { text: '.\nOtrā ', uncertain: false },
      { text: 'rinda', uncertain: true },
    ]);
    return editor;
  };

  it('renders uncertain words as highlights and keeps line breaks in the text', () => {
    const editor = setup();
    expect(editor.querySelectorAll('mark.uncertain')).toHaveLength(2);
    expect(extractText(editor)).toBe('Es redzu kaķi.\nOtrā rinda');
  });

  it('removes the highlight only from the word that was edited', () => {
    const editor = setup();
    editor.querySelector('mark')!.textContent = 'suni';
    expect(unwrapEditedMarks(editor)).toBe(true);
    expect(editor.querySelectorAll('mark.uncertain')).toHaveLength(1);
    expect(extractText(editor)).toBe('Es redzu suni.\nOtrā rinda');
  });

  it('leaves unedited highlights alone', () => {
    const editor = setup();
    expect(unwrapEditedMarks(editor)).toBe(false);
    expect(editor.querySelectorAll('mark.uncertain')).toHaveLength(2);
  });
});
