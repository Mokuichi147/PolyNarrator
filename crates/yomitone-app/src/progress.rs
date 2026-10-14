//! 進捗バーと状況の表示に使う、処理の進み具合

/// 章ごとに「話者の推測」と(音声を書き出す場合は)「音声合成」を 1 行ずつ進める
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Progress {
    /// 章ごとの行数
    counts: Vec<usize>,
    synthesize: bool,
    chapter: Option<usize>,
    estimated: usize,
    synthesized: usize,
}

impl Progress {
    pub fn new(counts: Vec<usize>, synthesize: bool) -> Self {
        Self { counts, synthesize, ..Self::default() }
    }

    pub fn chapter_started(&mut self, index: usize) {
        self.chapter = Some(index);
        self.estimated = 0;
        self.synthesized = 0;
    }

    pub fn sentence_estimated(&mut self) {
        self.estimated += 1;
    }

    pub fn synthesis_progressed(&mut self, done: usize) {
        self.synthesized = done;
    }

    fn units(&self, count: usize) -> usize {
        if self.synthesize {
            count * 2
        } else {
            count
        }
    }

    /// 0〜1 の進み具合
    pub fn fraction(&self) -> f64 {
        let total: usize = self.counts.iter().map(|&c| self.units(c)).sum();
        let Some(chapter) = self.chapter else {
            return 0.0;
        };
        if total == 0 {
            return 0.0;
        }
        let finished: usize = self.counts[..chapter.min(self.counts.len())].iter().map(|&c| self.units(c)).sum();
        let current = self.estimated + self.synthesized;
        ((finished + current) as f64 / total as f64).min(1.0)
    }

    /// 「2/5 章: 話者を推測しています (12/40 行)」のような今の状況
    pub fn status(&self) -> String {
        let Some(chapter) = self.chapter else {
            return "準備しています".into();
        };
        let count = self.counts.get(chapter).copied().unwrap_or(0);
        let position = format!("{}/{} 章", chapter + 1, self.counts.len());
        if self.synthesize && self.estimated >= count {
            format!("{position}: 音声を合成しています ({}/{count} 行)", self.synthesized)
        } else {
            format!("{position}: 話者を推測しています ({}/{count} 行)", self.estimated)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advances_through_estimation_and_synthesis() {
        let mut progress = Progress::new(vec![2, 2], true);
        assert_eq!(progress.fraction(), 0.0);
        assert_eq!(progress.status(), "準備しています");
        progress.chapter_started(0);
        progress.sentence_estimated();
        assert_eq!(progress.fraction(), 1.0 / 8.0);
        assert_eq!(progress.status(), "1/2 章: 話者を推測しています (1/2 行)");
        progress.sentence_estimated();
        progress.synthesis_progressed(1);
        assert_eq!(progress.fraction(), 3.0 / 8.0);
        assert_eq!(progress.status(), "1/2 章: 音声を合成しています (1/2 行)");
        progress.chapter_started(1);
        progress.sentence_estimated();
        assert_eq!(progress.fraction(), 5.0 / 8.0);
    }

    #[test]
    fn without_synthesis_counts_estimation_only() {
        let mut progress = Progress::new(vec![4, 0], false);
        progress.chapter_started(0);
        progress.sentence_estimated();
        assert_eq!(progress.fraction(), 0.25);
        progress.chapter_started(1);
        assert_eq!(progress.fraction(), 1.0);
        assert_eq!(progress.status(), "2/2 章: 話者を推測しています (0/0 行)");
    }
}
